require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')
require('rnn')

HierEpiModel = {}

include('util/read_data.lua')
include('util/Tree.lua')
include('util/Vocab.lua')
include('layers/CRowAddTable.lua')
include('models/TreeLSTM.lua')
include('models/ChildSumTreeLSTM.lua')
include('models/MemN2N.lua')
include('models/DMN.lua')
include('toefl/HEModelToefl.lua')
include('toefl/HierAttnModelToefl.lua')

printf = utils.printf

-- global paths (modify if desired)
HierEpiModel.data_dir       = 'data'
HierEpiModel.models_dir     = 'trained_models'

-- share module parameters
function share_params(cell, src)
  if torch.type(cell) == 'nn.gModule' then
    for i = 1, #cell.forwardnodes do
      local node = cell.forwardnodes[i]
      if node.data.module then
        if src.forwardnodes[i].data.module.bias ~= false then 
          node.data.module:share(src.forwardnodes[i].data.module,
          'weight', 'bias', 'gradWeight', 'gradBias')
        else
          node.data.module:share(src.forwardnodes[i].data.module,
          'weight', 'gradWeight', 'gradBias')
        end
      end
    end
  elseif torch.isTypeOf(cell, 'nn.Module') then
    cell:share(src, 'weight', 'bias', 'gradWeight', 'gradBias')
  else
    error('parameters cannot be shared for this input')
  end
end

function header(s)
  print(string.rep('-', 80))
  print(s)
  print(string.rep('-', 80))
end
