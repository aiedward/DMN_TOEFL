local DMNModelToefl = torch.class('HierAttnModel.DMNModelToefl')

function DMNModelToefl:__init(config)
  self.num_choices     = 4
  self.mem_dim        = config.mem_dim           or 75 --TreeLSTM memory dimension
  self.internal_dim   = config.internal_dim      or 75 --MemN2N memory dimension
  self.lr             = config.lr                or 1e-2
  self.batch_size     = config.batch_size        or 10
  self.reg            = config.reg               or 1e-4
  self.hops           = config.hops              or 1
  self.dropout        = config.dropout           or 0.0
  self.cuda           = config.cuda

  -- attention mechanism
  --self.sim = config.sim or 'dot'
  --self.att_norm = config.att_norm or 'sharp'

  -- word embedding
  self.emb_dim = config.emb_vecs:size(2)
  if self.cuda then
    self.emb = nn.LookupTable(config.emb_vecs:size(1), self.emb_dim):cuda()
    self.in_zeros = torch.zeros(self.emb_dim):cuda()
    self.emb.weight:copy(config.emb_vecs:cuda())
  else
    self.emb = nn.LookupTable(config.emb_vecs:size(1), self.emb_dim)
    self.in_zeros = torch.zeros(self.emb_dim)
    self.emb.weight:copy(config.emb_vecs)
  end

  -- optimizer configuration
  self.optimizer = config.optimizer or 'adagrad'
  self.optim_state = { learningRate = self.lr }

  -- negative log likelihood optimization objective
  if self.cuda then
    self.criterion = nn.DistKLDivCriterion():cuda()
  else
    self.criterion = nn.DistKLDivCriterion()
  end
  self.level = config.level or 'phrase'
  local treelstm_config = {
    in_dim = self.emb_dim,
    mem_dim = self.mem_dim,
    dropout = self.dropout,
    cuda = self.cuda
  }

  local dmn_config = {
    internal_dim = self.internal_dim,
    mem_dim = self.mem_dim,
    dropout = self.dropout,
    hops = self.hops,
    cuda = self.cuda
  }

  self.treelstm = HierAttnModel.ChildSumTreeLSTM(treelstm_config)
  self.DMN = HierAttnModel.DMN(dmn_config)
  self.sim_output_module = self:new_sim_module()

  local modules = nn.Parallel() -- to get the param in different module
  :add(self.treelstm)
  :add(self.DMN)
  :add(sim_output_module)
  self.params, self.grad_params = modules:getParameters()

end

function DMNModelToefl:new_sim_module()
  local choices = {}
  local similarities = {}

  local mem_out = nn.Identity()()
  for i=1,self.num_choices do
    choices[i] = nn.Identity()()
    similarities[i] = nn.CosineDistance(){mem_out,choices[i]}
  end
  local sim_all = nn.JoinTable(1)({similarities[1],similarities[2],similarities[3],similarities[4]})
  local sim_module = nn.gModule({mem_out,choices[1],choices[2],choices[3],choices[4]},{sim_all})

  local sim_output_module = nn.Sequential()
  :add(sim_module)
  :add(nn.LogSoftMax())

  if self.cuda then
    sim_output_module = sim_output_module:cuda()
  end

  return sim_output_module
end

function DMNModelToefl:train(dataset)

  --revoke sub module
  self.treelstm:training()
  self.DMN:training()
  local indices = torch.randperm(dataset.size) --shuffle
  --local tree_mem_zeros = torch.zeros(self.mem_dim)
  --local dmn_zeros = torch.zeros(self.internal_dim)
  --local avg_loss = 0
  --local count = 0
  for i = 1, dataset.size, self.batch_size do
    --count = count+1
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

    local feval = function(x)
      self.grad_params:zero()
      self.emb:zeroGradParameters()

      local loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]

        --query
        local emb_queries = {}
        local query_vecs = {}
        for q=1,#dataset.queries[idx] do
          self.emb:forward(dataset.queries[idx][q].sent)
          local emb_query
          if self.cuda then
            emb_query = torch.CudaTensor(self.emb.output:size()):copy(self.emb.output)
          else
            emb_query = torch.Tensor(self.emb.output:size()):copy(self.emb.output)
          end
          local _, query_vec = self.treelstm:unpack_state(self.treelstm:forward(dataset.queries[idx][q].tree.root, emb_query))
          table.insert(emb_queries,emb_query)
          table.insert(query_vecs,query_vec)
        end


        --choices
        local choice_vecs = {}
        local emb_choices = {}
        for c=1,self.num_choices do
          self.emb:forward(dataset.choices[idx][c].sent)
          local emb_choice
          if self.cuda then
            emb_choice = torch.CudaTensor(self.emb.output:size()):copy(self.emb.output)
          else
            emb_choice = torch.Tensor(self.emb.output:size()):copy(self.emb.output)
          end
          table.insert(emb_choices,emb_choice)
          local choice_root = dataset.choices[idx][c].tree.root
          local _, choice_vec = self.treelstm:unpack_state(self.treelstm:forward(choice_root, emb_choices[c]))
          table.insert(choice_vecs,choice_vec)
        end

        --memory network
        local memory
        local mem_fill_idx = 1

        local emb_sents  = {}
        for s=1,dataset.num_sent[idx] do
          local sent = dataset.sents[idx][s].sent
          local tree = dataset.sents[idx][s].tree
          self.emb:forward(sent)

          -- forward thru one tree
          local emb_sent
          if self.cuda then
            emb_sent = torch.CudaTensor(self.emb.output:size()):copy(self.emb.output)
          else
            emb_sent = torch.Tensor(self.emb.output:size()):copy(self.emb.output)
          end
          table.insert(emb_sents,emb_sent)

          self.treelstm:forward(dataset.sents[idx][s].tree.root,emb_sent)
          if self.level == 'phrase' then
            --collect hidden state (phrase level)
            for n=1,#tree.nodes do
              local c,h = self.treelstm:unpack_state(dataset.sents[idx][s].tree.nodes[n].state)
              if mem_fill_idx == 1 then
                memory = h
              else
                memory = torch.cat(memory, h, 2)
              end
              tree.nodes[n].memidx = mem_fill_idx
              mem_fill_idx = mem_fill_idx + 1
            end
          else
            -- collect hidden state (sentence level)
            local c,h = self.treelstm:unpack_state(dataset.sents[idx][s].tree.root.state)
            if mem_fill_idx == 1 then
              memory = h
            else
              memory = torch.cat(memory, h, 2)
            end
            mem_fill_idx = mem_fill_idx + 1
          end
        end
        if #memory:size() == 1 then
          memory = torch.reshape(memory,1,memory:size(1))
        else
          memory = memory:transpose(1,2)
        end

        --forward thru dmn
        local dmn_out = self.DMN:forward(memory,query_vecs,false,false)
        --calculate the possible choice
        pred = self.sim_output_module:forward{dmn_out,choice_vecs[1],choice_vecs[2],choice_vecs[3],choice_vecs[4]}
        loss = loss + self.criterion:forward(pred,dataset.golden[idx])
        
        local sim_grad = self.criterion:backward(pred,dataset.golden[idx])
        local mem_grad = self.sim_output_module:backward({dmn_out,choice_vecs[1],choice_vecs[2],choice_vecs[3],choice_vecs[4]},sim_grad)
        

        --backward thru choices
        for c=1,self.num_choices do
          self.treelstm:backward(dataset.choices[idx][c].tree,emb_choices[c],mem_grad[c+1],'choice') --mem_grad[1] is for memory network
        end

        --backward thru memory network
        mem_grad_input,query_grad = self.DMN:backward(memory,query_vecs,mem_grad[1])

        --backward thru query
        for q=1,#dataset.queries[idx] do
          self.treelstm:backward(dataset.queries[idx][q].tree,emb_queries[q],query_grad[q],'query')
        end

        --backward thru backround knowledge sent
        for s=1,dataset.num_sent[idx] do
          if self.level == 'phrase' then
            self.treelstm:backward(dataset.sents[idx][s].tree,emb_sents[s],mem_grad_input,'sent')
          else
            self.treelstm:backward(dataset.sents[idx][s].tree,emb_sents[s],mem_grad_input[s],'_sent') -- for sentence level
          end
        end
      end

      loss = loss / batch_size
      self.grad_params:div(batch_size)
      self.emb.gradWeight:div(batch_size)

      -- regularization
      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params)
      --avg_loss = avg_loss+loss
      return loss, self.grad_params
    end

    if self.optimizer == 'adagrad' then
      optim.adagrad(feval, self.params, self.optim_state)
    else
      optim.adam(feval, self.params, self.optim_state)
    end
  end
  xlua.progress(dataset.size, dataset.size)
  --avg_loss = avg_loss/count
  --return avg_loss
end

function DMNModelToefl:predict(container,sents,choices,query,num_sent,num_answers,verbose,test)
  --revoke sub module
  self.treelstm:evaluate()
  self.DMN:evaluate()
  local prediction = {}

  --query
  local query_vecs = {}
  if query ~= nil then
  for q=1,#query do
    self.emb:forward(query[q].sent)
    local emb_query
    if self.cuda then
      emb_query = torch.CudaTensor(self.emb.output:size()):copy(self.emb.output)
    else
      emb_query = torch.Tensor(self.emb.output:size()):copy(self.emb.output)
    end
    local query_root = query[q].tree.root
    local _, query_vec = self.treelstm:unpack_state(self.treelstm:forward(query_root, emb_query))
    table.insert(query_vecs,query_vec)
  end
  --choices
  local choice_vecs = {}
  local emb_choices = {}
  for c=1,self.num_choices do
    self.emb:forward(choices[c].sent)
    local emb_choice
    if self.cuda then
      emb_choice = torch.CudaTensor(self.emb.output:size()):copy(self.emb.output)
    else
      emb_choice = torch.Tensor(self.emb.output:size()):copy(self.emb.output)
    end
    table.insert(emb_choices,emb_choice)
    local choice_root = choices[c].tree.root
    local _, choice_vec = self.treelstm:unpack_state(self.treelstm:forward(choice_root, emb_choices[c]))
    table.insert(choice_vecs,choice_vec)
  end

  --memory network
  local memory
  local mem_fill_idx = 1

  local emb_sents  = {}

  if test == true then
    local A_file = (io.open(string.format("./attention%d.csv",self.hops), "a"))
    A_file:write(num_sent)
    A_file:write(',')
    A_file:close()
  end
  
  for s=1,num_sent do
    local sent = sents[s].sent
    local tree = sents[s].tree
    self.emb:forward(sent)

    -- forward thru one tree
    local emb_sent
    if self.cuda then
      emb_sent = torch.CudaTensor(self.emb.output:size()):copy(self.emb.output)
    else
      emb_sent = torch.Tensor(self.emb.output:size()):copy(self.emb.output)
    end
    table.insert(emb_sents,emb_sent)
    self.treelstm:forward(tree.root,emb_sent)
    if self.level == 'phrase' then
      -- collect hidden state (phrase level)
      for n=1,#tree.nodes do
        local c,h = self.treelstm:unpack_state(tree.nodes[n].state)
        if mem_fill_idx == 1 then
          memory = h
        else
          memory = torch.cat(memory, h,2)
        end
        mem_fill_idx = mem_fill_idx + 1
        tree.nodes[n].memidx = mem_fill_idx
      end
    else
      -- collect hidden state (sentence level)
      local c,h = self.treelstm:unpack_state(tree.root.state)
      if mem_fill_idx == 1 then
        memory = h
      else
        memory = torch.cat(memory, h, 2)
      end
      mem_fill_idx = mem_fill_idx + 1
    end
  end
  if #memory:size() == 1 then
    memory = torch.reshape(memory,1,memory:size(1))
  else
    memory = memory:transpose(1,2)
  end
  --forward thru dmn
  local mem_out = self.DMN:forward(memory,query_vecs,true,test)
  --export the vector to use tsne to print
  if test == true then
    local q_vv = query_vecs[1]
      if #query_vecs > 1 then
        for i=2,#query_vecs do 
          q_vv = torch.add(q_vv,1,query_vecs[i])
        end 
      end
    local out = (io.open(string.format("./vector%d.csv",self.hops), "a"))
    splitter = ","
    out:write('QUERY,')
    for i=1,q_vv:size(1) do
          out:write(string.format('%.4f',q_vv[i]))
          if i == q_vv:size(1) then
              out:write("\n")
          else
              out:write(splitter)
          end
    end
    out:write('CHOICES,')
    for i=1,#choice_vecs do 
      for j=1,choice_vecs[i]:size(1) do
        out:write(string.format('%.4f',choice_vecs[i][j]))
        if j== choice_vecs[i]:size(1) then
          out:write("\n")
        else
          out:write(splitter)
        end
      end
    end
    out:write('MEM_OUT,')
    for i=1,mem_out:size(1) do
          out:write(string.format('%.4f',mem_out[i]))
          if i == mem_out:size(1) then
              out:write("\n")
          else
              out:write(splitter)
          end
    end

    out:close()
    end
  --calculate the possible choice
  pred = self.sim_output_module:forward{mem_out,choice_vecs[1],choice_vecs[2],choice_vecs[3],choice_vecs[4]}

  prediction = argmax_n(pred,num_answers)

  --recycle! free the module
  for s=1,num_sent do
    self.treelstm:clean(sents[s].tree.root)
  end

  for q=1,#query do
    self.treelstm:clean(query[q].tree.root)
  end

  for c=1,self.num_choices do
    self.treelstm:clean(choices[c].tree.root)
  end
  self.DMN:clean()
  end
  return prediction
end

function DMNModelToefl:predict_dataset(dataset,test)

  local predictions = {}
  for i = 1, dataset.size do
    xlua.progress(i, dataset.size)
    if test == true then  
      local A_file = (io.open(string.format("./attention%d.csv",self.hops), "a"))
      A_file:write(i)
      A_file:write(',')
      A_file:close()
    end
    table.insert(predictions,self:predict(dataset.container[i],dataset.sents[i],dataset.choices[i],dataset.queries[i],dataset.num_sent[i],#dataset.answers[i],i==1,test))
  end
  return predictions
end

function argmax_n(v,n)
  sorted,indices = torch.sort(v,true)
  ret , _ = torch.sort(indices[{{1,n}}])
  return ret
end

function argmax(v)
  local idx = 1
  local max = v[1]
  for i = 2, v:size(1) do
    if v[i] > max then
      max = v[i]
      idx = i
    end
  end
  return idx
end

function DMNModelToefl:print_config()
  local num_params = self.params:size(1)
  printf('%-25s = %d\n',   'num params', num_params)
  printf('%-25s = %d\n',   'Number of choices', self.num_choices)
  printf('%-25s = %d\n',   'Tree-LSTM memory dim', self.mem_dim)
  printf('%-25s = %d\n',   'DMN dim', self.internal_dim)
  --printf('%-25s = %d\n',   'memory size', self.mem_size)
  printf('%-25s = %.2e\n', 'learning rate', self.lr)
  printf('%-25s = %d\n',   'minibatch size', self.batch_size)
  printf('%-25s = %.2e\n', 'regularization strength', self.reg)
  printf('%-25s = %d\n',   'hops', self.hops)
  printf('%-25s = %s\n',   'dropout', tostring(self.dropout))
  --printf('%-25s = %s\n',   'attention similarity', self.sim)
  --printf('%-25s = %s\n',   'attention normalization', self.att_norm)
  printf('%-25s = %d\n',   'word vector dim', self.emb_dim)
  printf('%-25s = %s\n',   'optimizer', self.optimizer)
end

function DMNModelToefl:save(path)
    local config = {
    level                   = self.level,
		num_choices             = self.num_choices,
		internal_dim            = self.internal_dim,
    mem_dim                 = self.mem_dim,
        --mem_size                = self.mem_size,
		lr                      = self.lr,
		batch_size              = self.batch_size,
		reg                     = self.reg,
		hops                    = self.hops,
    dropout                 = self.dropout,
		sim                     = self.sim,
		att_norm                = self.att_norm,
    emb_vecs                = self.emb.weight:float(),
    optimizer               = self.optimizer,
    cuda                    = self.cuda
    }

    torch.save(path, {
        params = self.params,
        config = config,
    })
end


function DMNModelToefl.load(path)
  local state = torch.load(path)
  local model = HierAttnModel.DMNModelToefl.new(state.config)
  model.params:copy(state.params)
  return model
end
