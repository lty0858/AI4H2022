# PILLOW
- https://pypi.org/project/Pillow/
- [Tutorial - Pillow (PIL Fork) 9.4.0 documentation](https://pillow.readthedocs.io/en/stable/handbook/tutorial.html)
- [Day24-Python 影像處理套件 PIL](https://ithelp.ithome.com.tw/articles/10226578)
- [[Day 21] 從零開始學Python - 基本圖形處理Pillow：花下是誰對影成雙](https://ithelp.ithome.com.tw/articles/10247292)
- [Python Pillow Tutorial - GeeksforGeeks](https://www.geeksforgeeks.org/python-pillow-tutorial/)
- [Python Pillow Tutorial - Javatpoint](https://www.javatpoint.com/python-pillow)
##
```
from PIL import Image
img = Image.open("img01.jpg")
img.show()
w,h=img.size
print(w,h) #320 240
filename=img.filename
print(filename) #"media\img01.jpg
```
