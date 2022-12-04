# TiBox2d
A Taichi Hackthon project  
Yet another game-oriented GPU Physics Engine

## Motivation: 
1. Fulfill a wish from 2 years ago when I first learned Taichi and GAMES201
完成一个鸽了许久的承诺
2. Most developers who want to write a physics simulation game with Taichi can't help but use "MPM99" as a template and modify it, so it's time to bring some diversity.
写游戏竟然不用PBD？
3. for fun
确实很好玩

## API与使用方法
故事开始
``` python
import TiBox2d
import taichi as ti
```
创建物Tibox2d理收集器
``` python
tibox_collector = TiBox2d.Collector()
```
使用收集器收集你想模拟的物理材料，输入它AABB左下和右上的位置坐标
```python
tibox_collector.add_water_box(5.0, 5.0, 15.0, 15.0)
tibox_collector.add_plastic_box(16.0, 20.0, 26.0, 30.0)
tibox_collector.add_rigid_box(27.0, 35.0, 37.0, 45.0)
tibox_collector.add_elastic_box(38.0, 50.0, 48.0, 60.0)
tibox_collector.add_rope(50.0, 0.0, 30.0)
```
创建求解器并编译
```python
tibox_solver = TiBox2d.Solver(...)
tibox_solver.Compile(tibox_collector)
```
在主循环中调用step进行模拟
```python
tibox_solver = TiBox2d.Solver(...)
tibox_solver.step(tibox_collector)
```
这时位置就会在```tibox_solver.positions```中得到更新

### Demo1: Frozen19 (中文：19行代码的《冰雪奇缘》)
```python
cd src
python3 frozen19.py
```

### Demo2(Incomplete): AngryTi中文：愤怒的小鸟太极版)
```python
cd src
python3 AngryTi.py
```

## Todo List (Priority from highest to lowest):
- [x] rigid Body  (shape matching)
- [x] elastic body  (region based shape matching)
- [x] plastic body fake  (shape matching with some trick)
- [x] plastic body real  (shape matching with yield&creep)
- [x] Neighborhood Search for collision handling
- [x] ropes  (Follow the leader & shape matching)
- [x] fluid  (PBF)
- [ ] cloth
