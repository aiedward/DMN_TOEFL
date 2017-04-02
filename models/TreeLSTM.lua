--[[

  Tree-LSTM base class

--]]

local TreeLSTM, parent = torch.class('HierAttnModel.TreeLSTM', 'nn.Module')

function TreeLSTM:__init(config)
    parent.__init(self)
    self.in_dim = config.in_dim
    if self.in_dim == nil then error('input dimension must be specified') end
    self.mem_dim = config.mem_dim or 150
    self.dropout = config.dropout or 0.0
    self.train = false
end

function TreeLSTM:training()
    self.train = true
end

function TreeLSTM:evaluate()
    self.train = false
end

function TreeLSTM:allocate_module(tree, module)
    local modules = module .. 's'
    --print(modules)
    local num_free = #self[modules]
    if num_free == 0 then
        tree[module] = self['new_' .. module](self)
    else
        tree[module] = self[modules][num_free]
        self[modules][num_free] = nil
    end

    -- necessary for dropout to behave properly
    if self.train then tree[module]:training() else tree[module]:evaluate() end
end

function TreeLSTM:free_module(tree, module)
    if tree[module] == nil then return end
    table.insert(self[module .. 's'], tree[module])
    tree[module] = nil
end
