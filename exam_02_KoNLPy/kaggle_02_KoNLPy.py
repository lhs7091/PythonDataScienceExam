'''
refer to inflearn.com/course/데이터-사이언트-kaggle/lecture/11348
Initial set up konlpy

pip install JPype1-py3
pip install konlpy
'''
from konlpy.tag import Kkma
from konlpy.utils import pprint
kkma = Kkma()
print(kkma.sentences(u'一生懸命しましょう。。'))
print(kkma.nouns(u'一生懸命しましょう。。'))
print(kkma.pos(u'一生懸命しましょう。。'))

'''
RESULT
['一生懸命しましょう。。']
[]
[('一生懸命', 'OH'), ('し', 'SW'), ('ま', 'SW'), ('し', 'SW'), ('ょ', 'SW'), ('う', 'SW'), ('。。', 'SW')]

'''