
for (( i = 1; i<=5; i=i+1))
do
  th toefl/main.lua  --level sentence  --hops 1 --prune 1 --dropout 0.5 --dim 75 --internal 75  
  th toefl/main.lua  --level sentence  --hops 2 --prune 1 --dropout 0.5 --dim 75 --internal 75 
done
for (( i = 1; i<=5; i=i+1))
do
  th toefl/main.lua  --level sentence  --hops 1 --prune 0.5 --dropout 0.5 --dim 75 --internal 75 
  th toefl/main.lua  --level sentence  --hops 2 --prune 0.5 --dropout 0.5 --dim 75 --internal 75 
done
for (( i = 1; i<=5; i=i+1))
do
  th toefl/main.lua  --level sentence  --hops 1 --prune 0.1 --dropout 0.5 --dim 75 --internal 75 
  th toefl/main.lua  --level sentence  --hops 2 --prune 0.1 --dropout 0.5 --dim 75 --internal 75 
done
for (( i = 1; i<=5; i=i+1))
do
  th toefl/main.lua  --level phrase  --hops 1 --prune 1 --dropout 0.5 --dim 75 --internal 75 
  th toefl/main.lua  --level phrase  --hops 2 --prune 1 --dropout 0.5 --dim 75 --internal 75 
done
