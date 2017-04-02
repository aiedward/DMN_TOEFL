require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')
require('rnn')

HierAttnModel = {}

include('util/read_data.lua')
include('util/Tree.lua')
include('util/Vocab.lua')
include('layers/CRowAddTable.lua')
include('models/TreeLSTM.lua')
include('models/ChildSumTreeLSTM.lua')
include('models/DMN.lua')
include('models/DMN_liu.lua')
include('toefl/TreeLSTMToefl.lua')
include('toefl/DMNModelToefl.lua')
include('toefl/DMNModelToefl_liu.lua')

printf = utils.printf

-- global paths (modify if desired)
HierAttnModel.data_dir       = 'data'
HierAttnModel.models_dir     = 'trained_models'

-- share module parameters
function share_params(cell, src)
  if torch.type(cell) == 'nn.gModule' then
    --print(#cell.forwardnodes)
    --print(#src.forwardnodes)
    for i = 1, #cell.forwardnodes do
      --if i==20 then
        --print(i)
        --print({cell.forwardnodes[i]})
      --end
      local node = cell.forwardnodes[i]
      if node.data.module then
        --if i==20 or i==19 or i==18 or i==21 then
          --print(i)
          --print({src.forwardnodes[i].data.module})
        --end
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
