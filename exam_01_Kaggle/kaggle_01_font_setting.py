'''
refer to inflearn.com/course/데이터-사이언트-kaggle/lecture/11348

setting for drawing graph on notebook
%matplotlib inline
'''
import warnings
warnings.filterwarnings("ignore")

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# for minus number on graph
mpl.rcParams['axes.unicode_minus'] = False

# construct a temporary data for drawing graph
import numpy as np
data = np.random.randint(-100, 100, 50).cumsum()
print(data)

plt.plot(range(50), data, 'r')
plt.title('시간별 가격 추이')
plt.ylabel('주식가격')
plt.xlabel('시간(분)')
plt.show() # error of korean font occurred

# check the matplotlib's path and version information
print('version: ', mpl.__version__)
print('path: ', mpl.__file__)
print('setting: ', mpl.get_configdir())
print('cache: ', mpl.get_cachedir())

# move the path of setting files directory by terminal
print('setting files path:', mpl.matplotlib_fname())

# number of all of fonts
font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
print(len(font_list)) # total 328
# for mac
font_list_mac = fm.OSXInstalledFonts()
print(len(font_list_mac))

[(f.name, f.fname) for f in fm.fontManager.ttflist if 'Nanum' in f.name]

'''
1) Font Properties
    the way using by fname option
'''
path = '/Library/Fonts/NanumGothicExtraBold.otf'
fontprop = fm.FontProperties(fname=path, size=16)

plt.plot(range(50), data, 'r')
plt.title('시간별 가격 추이', fontproperties=fontprop)
plt.ylabel('주식가격', fontproperties=fontprop)
plt.xlabel('시간(분)', fontproperties=fontprop)
plt.show() # error of korean font occurred

'''
2) matplotlib.rcParams[]
    the way using by matplotlib.rcParams[]
'''
# default settings
print('# default font size')
print(plt.rcParams['font.size'])
print('# default font ')
print(plt.rcParams['font.family'])
'''
setting the font we want
if it's not read, you have to be remove fontlist-v310.json
and restart
'''
plt.rcParams["font.family"] = 'NanumGothicOTF'
plt.rcParams["font.size"] = 20
plt.rcParams["figure.figsize"] = (14,4)

plt.plot(range(50), data, 'r')
plt.title('시간별 가격 추이')
plt.ylabel('주식가격')
plt.xlabel('시간(분)')
plt.style.use('seaborn-pastel')
plt.show() # error of korean font occurred

'''
3) The way using by FontProperties and plt.rc instead of rcParams
'''
path = '/Library/Fonts/NanumGothicExtraBold.otf'
font_name = fm.FontProperties(fname=path, size=50).get_name()
print(font_name)
plt.rc('font', family=font_name)

fig, ax = plt.subplots()
ax.plot(data)
ax.set_title('시간별 가격 추이')
plt.ylabel('주식가격')
plt.xlabel('시간(분)')
plt.style.use('ggplot')
plt.show()

'''
4) if you want change permanent, 
setting file of 'matplotlibrc' should be changed
about 199 lines font.family = *****
'''

fig, ax = plt.subplots()
ax.plot(10*np.random.randn(100),10*np.random.randn(100), 'o')
ax.set_title('숫자 분포도 보기')
plt.show()
