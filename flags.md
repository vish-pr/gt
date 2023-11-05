# DEBUG
* example: DEBUG=1 python3  tiny_stories/model.py
* =1
* =0
* =2
* =3



# GRAPH



# prof
```
python3 -m cProfile -o out.prof tiny_stories/model.py --count=10
snakeviz out.prof
```