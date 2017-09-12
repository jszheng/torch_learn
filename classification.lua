require 'torch';
-- require 'itorch';
require 'image';

local image_url = 'http://upload.wikimedia.org/wikipedia/commons/e/e9/Goldfish3.jpg'
local network_url = 'https://www.dropbox.com/s/npmr5egvjbg7ovb/nin_nobn_final.t7'
image_name = paths.basename(image_url)
network_name = paths.basename(network_url)
if not paths.filep(image_name) then os.execute('wget '..image_url)   end
if not paths.filep(network_name) then os.execute('wget '..network_url)   end

net = torch.load(network_name):unpack():float()
net:evaluate()
print(tostring(net))

-- itorch.image(net:get(1).weight)

im = image.load(image_name)
--itorch.image(image.scale(im, 256, 256)) -- rescale just to show the image

-- Rescales and normalizes the image
function preprocess(im, img_mean)
  -- rescale the image
  local im3 = image.scale(im,224,224,'bilinear')
  -- subtract imagenet mean and divide by std
  for i=1,3 do im3[i]:add(-img_mean.mean[i]):div(img_mean.std[i]) end
  return im3
end

I = preprocess(im, net.transform):float()
--itorch.image(I)

synset_words = {}
for line in io.lines'7_imagenet_classification/synset_words.txt' do
    table.insert(synset_words, line:sub(11))
end

local _,classes = net:forward(I):view(-1):sort(true)
for i=1,5 do
  print('predicted class '..tostring(i)..': ', synset_words[classes[i] ])
end
