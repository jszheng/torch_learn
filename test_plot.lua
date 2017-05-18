require 'torch'
require 'gnuplot'

x=torch.linspace(-2*math.pi,2*math.pi)
gnuplot.plot(torch.sin(x))

x = torch.linspace(-1,1)
xx = torch.Tensor(x:size(1),x:size(1)):zero():addr(1,x,x)
xx = xx*math.pi*6
gnuplot.splot(torch.sin(xx))