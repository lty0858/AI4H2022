# PILLOW
- https://pypi.org/project/Pillow/
- [Python Pillow Tutorial - GeeksforGeeks](https://www.geeksforgeeks.org/python-pillow-tutorial/)
- [Tutorial - Pillow (PIL Fork) 9.4.0 documentation](https://pillow.readthedocs.io/en/stable/handbook/tutorial.html)

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
