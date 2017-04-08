local DMN, parent = torch.class('HierEpiModel.DMN','nn.Module')

function DMN:__init(config)
  parent.__init(self)
  self.mem_dim = config.mem_dim
  self.internal_dim = config.internal_dim
  self.hops = config.hops
  self.train = false
  self.dropout = config.dropout
  self.cuda = config.cuda
  self.Mem_DNNs = {}
  self.Attn_GRUs = {}
  self.attn_gates = {}
  self.Soft_Attns = {}
  --Creat Two master module & #hops DNN modules--
  self.master_attn_gate = self:new_attn_gate()
  self.master_Attn_GRU = self:new_Attn_GRU()
  for i=1,self.hops do
    self.Mem_DNNs[i] = self:new_Mem_DNN()
  end
  self.depth =0
  -- Hidden State initial values-----------------
  local initial
  if self.cuda then
    initial = torch.zeros(self.internal_dim):cuda()
  else 
    initial = torch.zeros(self.internal_dim)
  end
  self.h_inital = initial
  ------------------------------------------------
end
--Inputs:{f:Tensor,q_in:Table,m_p:Tensor}--Ouputs:Tensor(size1)
function DMN:new_attn_gate()
  local f = nn.Identity()()
  local q_in = nn.Identity()()
  local m_p = nn.Identity()()
  local q = nn.CAddTable()(q_in)
  local z = {}
  z[1] = nn.CMulTable(){f,q}
  z[2] = nn.CMulTable(){f,m_p}
  z[3] = nn.Abs()(nn.CSubTable(){f,q})
  z[4] = nn.Abs()(nn.CSubTable(){f,m_p})
  local Z = nn.Linear(self.mem_dim,1)(nn.Tanh()
  (nn.Linear(#z*self.mem_dim,self.mem_dim)(nn.JoinTable(1)(z))))
  local attn_gate = nn.gModule({f,q_in,m_p},{Z})
  if self.cuda then
    attn_gate = attn_gate:cuda()
  end
  if self.master_attn_gate ~= nil then
    share_params(attn_gate,self.master_attn_gate)
  end
  return attn_gate
end
--Inputs:{inputs:Table},outputs:Tensor
function DMN:new_Soft_Attn()
  local inputs = nn.Identity()()
  local con = nn.JoinTable(1)(inputs)
  local output = nn.SoftMax()(con)
  local Soft_Attn = nn.gModule({inputs},{output})
  if self.cuda then
    Soft_Attn = Soft_Attn:cuda()
  end
  return Soft_Attn
end
--Inputs:f:Tensor,h_p:Tensor,gate:Tensor(1) Ouputs:h:Tensor
function DMN:new_Attn_GRU()
  local f = nn.Identity()()
  local h_p = nn.Identity()()
  local gate = nn.Identity()() 
  --GRU
  local r = nn.Sigmoid()(
  nn.CAddTable(){
    nn.Dropout(self.dropout)(nn.Linear(self.mem_dim,self.internal_dim)(f)),
    nn.Dropout(self.dropout)(nn.Linear(self.internal_dim,self.internal_dim)(h_p))
  })
  local U = nn.Linear(self.internal_dim,self.internal_dim)
  U.bias=false
  local Uh = nn.Dropout(self.dropout)(U(h_p))
  local _h = nn.Tanh()(
  nn.CAddTable(){
    nn.Dropout(self.dropout)(nn.Linear(self.mem_dim,self.internal_dim)(f)),
    nn.CMulTable(){r,Uh}
  }
  )
  local attn_gate = nn.Sum(2)(nn.Replicate(self.internal_dim)(gate))
  local h = nn.CAddTable(){ 
    nn.CMulTable(){attn_gate,_h},
    nn.CMulTable(){ 
      nn.SAdd(-1,true)(attn_gate),h_p} 
    }
    ---------------------------------------
    local Attn_GRU = nn.gModule({f,h_p,gate},{h})
    if self.cuda then Attn_GRU = Attn_GRU:cuda()
    end
    if self.master_Attn_GRU ~= nil then
      share_params(Attn_GRU, self.master_Attn_GRU)
    end
    return Attn_GRU
end

function DMN:new_Mem_DNN()
    local m_p = nn.Identity()()
    local q_in = nn.Identity()()
    local e = nn.Identity()()
    local q = nn.CAddTable()(q_in)
    local m = (nn.Linear((2*self.mem_dim+self.internal_dim),self.mem_dim)(nn.JoinTable(1)({m_p,e,q})))
    local Mem_DNN = nn.gModule({m_p,e,q_in},{m})
    if self.cuda then Mem_DNN = Mem_DNN:cuda()
    end
    -- Do not share parameters
    return Mem_DNN
end

  function DMN:forward(facts,query)
    -- facts is a Matrix tensor, query is a table
    local T = facts:size(1) -- time steps
    -- sum the tensor inside the query table 
    local q_prev = query[1]
    if #query > 1 then
      for i=2,#query do 
        q_prev = torch.add(q_prev,1,query[i])
      end 
    end 
    --
    for i = 1, self.hops do
      local m_p --previous memory
      if i == 1 then m_p = q_prev  --origin m_p = query
      else m_p = self.Mem_DNNs[i-1].output end
      --generate attention--
      local gates_hop = self.attn_gates[i]
      if gates_hop == nil then
        gates_hop = {}
        self.attn_gates[i] = gates_hop
      end
      local g={}
      for t = 1, T do 
        local attn_gate = self.attn_gates[i][t]
        if attn_gate == nil then
          attn_gate = self:new_attn_gate()
          self.attn_gates[i][t] = attn_gate
        end
        g[t] = attn_gate:forward({facts[t],query,m_p})
      end
      -------
      local Soft_Attn = self.Soft_Attns[i]
      if Soft_Attn == nil then
        Soft_Attn = self:new_Soft_Attn() --new Soft_Attn module
        self.Soft_Attns[i] = Soft_Attn
      end
      local Attn_Vector = Soft_Attn:forward(g)
      
      --Start run GRU to generate episode
      self.depth = 0
      local episode
      --- new one {} in Attn_GRUs
      local Attn_Hop = self.Attn_GRUs[i]
      if Attn_Hop == nil then
        Attn_Hop = {}
        self.Attn_GRUs[i] = Attn_Hop
      end
      --- 
      for t =1,T do
        self.depth = self.depth+1
        -- new a Attn_GRU module in {}
        local Attn_GRU = self.Attn_GRUs[i][self.depth]
        if Attn_GRU == nil then
          Attn_GRU = self:new_Attn_GRU()
          self.Attn_GRUs[i][self.depth] = Attn_GRU
        end

        local prev_h
        if self.depth >1 then prev_h = self.Attn_GRUs[i][self.depth-1].output
        else prev_h = self.h_inital  end

        episode= Attn_GRU:forward({facts[t],prev_h,Attn_Vector[t]*torch.ones(1)})
      end
      local Mem_DNN = self.Mem_DNNs[i]  --new Mem_DNN module 
      self.output = Mem_DNN:forward({m_p,episode,query})

    end
    return self.output
  end

  function DMN:backward(facts,query,grad_outputs)
    local grad_facts
    local grad_query={}
    if self.cuda then
      grad_facts = torch.CudaTensor(facts:size())
    else
      grad_facts = torch.Tensor(facts:size())
    end
    grad_facts:zero()

    local T = facts:size(1) --time step T
    if self.depth ==0 then
      error("No cells to backpropagate through")
    end

    local q_prev = query[1]
    if #query > 1 then
      for i=2,#query do 
        q_prev = torch.add(q_prev,1,query[i])
      end
    end 
    -------------------------------------------- 

    local memory_grad
    local attn_memory_grad
    for i = self.hops,1,-1 do 
      
      local m_p = (i > 1) and self.Mem_DNNs[i-1].output or q_prev --origin = query
      
      local depth = #self.Attn_GRUs[i]
      local episode = self.Attn_GRUs[i][depth].output
      local Mem_DNN = self.Mem_DNNs[i] -- take out the DNN 
      -- decide the gradoutput 
      if i == self.hops then memory_grad = grad_outputs
      -- ayou7995 : strange for add self.attn_gates[i][t] to memory_grad ??????
      else 
        memory_grad = self.Mem_DNNs[i+1].gradInput[1]+attn_memory_grad
      end
      -- backward
      Mem_DNN:backward({m_p,episode,query}, memory_grad)
      if i==self.hops then
        grad_query = Mem_DNN.gradInput[3]
      else
        for k=1,#grad_query do
          grad_query[k]:add(1,Mem_DNN.gradInput[3][k])
        end
      end

      local g_i = self.Soft_Attns[i].output --Tensor
      local grad_g ={}
      local input_z = {}
      for t = T ,1,-1 do 
        local Attn_GRU = self.Attn_GRUs[i][t]
        local h_p = (t>1) and self.Attn_GRUs[i][t-1].output or self.h_inital
        local h_grad 
        if t == T then 
          h_grad = Mem_DNN.gradInput[2]
        else 
          h_grad = self.Attn_GRUs[i][t+1].gradInput[2]
        end

        Attn_GRU:backward({facts[t],h_p,g_i[t]},h_grad)
        grad_g[t] = Attn_GRU.gradInput[3]
        grad_facts[t]:add(Attn_GRU.gradInput[1])
        --for SoftMax inputs
        input_z[t] = self.attn_gates[i][t].output
      end

      local grad_Attn_Vector = torch.Tensor(#grad_g)
      for k=1,T do 
        grad_Attn_Vector[k] = grad_g[k][1]
      end

      local grad_z = self.Soft_Attns[i]:backward(input_z,grad_Attn_Vector)
      
      if self.cuda then
        attn_memory_grad = torch.zeros(self.mem_dim):cuda()
      else
        attn_memory_grad = torch.zeros(self.mem_dim)
      end
      for t = T,1,-1 do 
        local attn_gate = self.attn_gates[i][t]
        attn_gate:backward({facts[t],query,m_p},grad_z[t])
        attn_memory_grad:add(attn_gate.gradInput[3])
        grad_facts[t]:add(attn_gate.gradInput[1])  

        for k=1,#grad_query do
          grad_query[k]:add(attn_gate.gradInput[2][k])
        end
      end
    end
    return grad_facts, grad_query
  end

  function DMN:training()
    self.train = true
  end

  function DMN:evaluate()
    self.train = false
  end

  function DMN:parameters()
    local params, grad_params = {}, {}
    local dnn_p, dnn_g = {}, {}
    --local soft_p, soft_g = {}, {}
    for idx = 1, #self.Mem_DNNs do
      local d_p, d_g = self.Mem_DNNs[idx]:parameters()
      tablex.insertvalues(dnn_p, d_p)
      tablex.insertvalues(dnn_g, d_g)
    end
    --for idx = 1, #Soft_Attns do
    --local s_p, s_g = Soft_Attns[idx]:paramters()
    --tablex.insertvalues(soft_p, s_p)
    --tablex.insertvalues(soft_g, s_g)
    --end
    local attn_gate_p, attn_gate_g = self.master_attn_gate:parameters()
    local attn_gru_p, attn_gru_g = self.master_Attn_GRU:parameters()
    tablex.insertvalues(params,dnn_p)
    tablex.insertvalues(params,attn_gate_p)
    tablex.insertvalues(params,attn_gru_p)
    --tablex.insertvalues(params,soft_p)
    tablex.insertvalues(grad_params,dnn_g)
    tablex.insertvalues(grad_params,attn_gate_g)
    tablex.insertvalues(grad_params,attn_gru_g)
    --tablex.insertvalues(grad_params,soft_g)
    return params, grad_params
  end

  function DMN:clean()  

    self.Attn_GRUs = nil
    self.Attn_GRUs = {}
    self.attn_gates = nil
    self.attn_gates = {}
    self.Soft_Attns = nil
    self.Soft_Attns = {}
    collectgarbage()
  end

