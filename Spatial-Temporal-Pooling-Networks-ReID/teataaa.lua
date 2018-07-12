matio = require ‘matio’


local filename = '/home/liu/Dataset/i-LIDS-VID-HEAT/cam1/person001/cam1_person001_00317.mat'

local a = matio.load(filename)

print(a)