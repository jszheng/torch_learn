local torch = require 'torch'

local data = torch.Tensor{
  {1, 2, 3},
  {4, 5, 6},
  {7, 8, 9},
}

print(data)

torch.manualSeed(1234)
-- choose a dimension
N = 5

-- create a random NxN matrix
A = torch.rand(N, N)
print(A)

-- make it symmetric positive
A = A*A:t()

-- make it definite
A:add(0.001, torch.eye(N))

-- add a linear term
b = torch.rand(N)

-- create the quadratic form
function J(x)
   return 0.5*x:dot(A*x)-b:dot(x)
end

print(J(torch.rand(N)))

xs = torch.inverse(A)*b
print(string.format('J(x^*) = %g', J(xs)))

function dJ(x)
  return A*x-b
end

x = torch.rand(N)
lr = 0.01
for i=1,20000 do
  x = x - dJ(x)*lr
  -- we print the value of the objective function at each iteration
  print(string.format('at iter %d J(x) = %f', i, J(x)))
end
local A = torch.rand(N, N)
