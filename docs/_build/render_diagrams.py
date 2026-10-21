from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'assets'
ASSETS.mkdir(exist_ok=True)
REG = 'C:/Windows/Fonts/msyh.ttc'
BOLD = 'C:/Windows/Fonts/msyhbd.ttc'
BG = '#F3F7F8'
INK = '#163C3A'
MUTED = '#50696D'
TEAL = '#187E75'
BLUE = '#356DA5'
AMBER = '#A56C1A'
RED = '#AF5757'

def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else REG, size)

def wrapped(draw, text, xy, width, size=26, fill=MUTED, bold=False, gap=10):
    f = font(size, bold)
    y = xy[1]
    for para in text.split('\n'):
        line = ''
        for char in para:
            if draw.textlength(line + char, font=f) > width and line:
                draw.text((xy[0], y), line, font=f, fill=fill)
                y += size + gap
                line = char
            else:
                line += char
        draw.text((xy[0], y), line, font=f, fill=fill)
        y += size + gap
    return y

def arrow(draw, points, color=TEAL, width=5, dashed=False):
    if dashed:
        for a, b in zip(points, points[1:]):
            length = math.dist(a,b)
            for i in range(0, int(length), 22):
                s, e = i/length, min(i+12,length)/length
                draw.line([(a[0]+(b[0]-a[0])*s,a[1]+(b[1]-a[1])*s),
                           (a[0]+(b[0]-a[0])*e,a[1]+(b[1]-a[1])*e)], fill=color,width=width)
    else:
        draw.line(points, fill=color, width=width, joint='curve')
    a,b = points[-2:]
    angle=math.atan2(b[1]-a[1],b[0]-a[0])
    head=[b,(b[0]-17*math.cos(angle-.5),b[1]-17*math.sin(angle-.5)),
          (b[0]-17*math.cos(angle+.5),b[1]-17*math.sin(angle+.5))]
    draw.polygon(head,fill=color)

def note(draw, x,y,w,h,title,body,color=BLUE):
    draw.rounded_rectangle((x,y,x+w,y+h), radius=22, fill='white', outline='#D5E2E3', width=2)
    draw.rounded_rectangle((x,y,x+8,y+h),radius=4,fill=color)
    after=wrapped(draw,title,(x+27,y+23),w-54,30,color,True)
    end=wrapped(draw,body,(x+27,after+17),w-54,25)
    assert end <= y+h-5, (title,end,y+h)

def header(draw, title, subtitle, width):
    draw.rectangle((0,0,width,16), fill=TEAL)
    draw.text((65,54),title,font=font(48,True),fill=INK)
    wrapped(draw,subtitle,(68,129),width-136,27)

# Diagram 1: full existing workflow, including optional checks and review loops.
im=Image.new('RGB',(1800,2220),BG)
d=ImageDraw.Draw(im)
header(d,'一段机器人录制，如何变成训练数据','完整业务流程  ·  对应公司平台的主要入口  ·  2026.10.15',1800)
nodes=[
    ('录制','画面＋机器人状态＋动作','位置：机器人采集端'),
    ('上传','原始文件进入统一存储','入口：上传列表'),
    ('准备数据','按格式转化，或直接读取','入口：数据转化／直接读取路径'),
    ('扫描登记','登记编号、时长、相机和文件位置','入口：Episodes 管理'),
    ('标注与动作划分','划分大任务／子任务，填写起止位置','入口：我的任务 → 标注编辑器'),
    ('审核','检查名称、边界、遗漏与有效性','入口：审核队列 → 审核队列标签'),
    ('验收','按交付要求作最终确认','入口：审核队列 → 验收队列标签'),
    ('导出训练集','按整段／大任务／子任务组织数据','入口：数据导出 → LeRobot 训练集'),
    ('模型训练','训练程序读取画面、动作与指令','位置：后续训练系统'),
]
x,w,h,top,step=550,700,146,248,202
for i,(title,body,entry) in enumerate(nodes):
    y=top+i*step
    d.rounded_rectangle((x,y,x+w,y+h),radius=22,fill='white',outline=TEAL if i in (4,5,6) else '#C9DBDC',width=3)
    d.ellipse((x+22,y+23,x+76,y+77),fill=TEAL)
    d.text((x+37,y+29),str(i+1),font=font(30,True),fill='white')
    d.text((x+96,y+19),title,font=font(34,True),fill=INK)
    d.text((x+96,y+70),body,font=font(26),fill=MUTED)
    d.text((x+96,y+110),entry,font=font(23),fill=TEAL)
    if i<8:
        arrow(d,[(x+w/2,y+h+5),(x+w/2,y+step-8)])

note(d,65,248,420,300,'采集前先准备','选择项目空间、机器人配置、机器编号和采集人员。\nEpisode＝一次录制；可能包含等待、重试或多个任务。')
note(d,1330,652,400,285,'基础质检','按配置检查文件缺失、断流、频率等问题。\n结果提供给后续处理人员；当前相关路径属于信息提示。',AMBER)
arrow(d,[(1255,725),(1320,725)],AMBER,dashed=True)
note(d,1330,1056,400,300,'AI 质量分析','平台入口：采集数据质量。\n可另行发起分析，查看疑似问题片段。是否可运行取决于数据与模型就绪情况。',AMBER)
arrow(d,[(1255,927),(1292,927),(1292,1129),(1320,1129)],AMBER,dashed=True)
note(d,65,1056,420,245,'动作划分是什么','给长录像加“目录”：\n哪一段在取杯，哪一段在加料，何时开始、何时结束。',TEAL)
# Review / acceptance returns to annotation, routed outside the central lane.
arrow(d,[(545,top+6*step+73),(515,top+6*step+73),(515,top+4*step+73),(542,top+4*step+73)],RED,4)
arrow(d,[(545,top+5*step+73),(515,top+5*step+73)],RED,4)
wrapped(d,'审核／验收有问题\n退回标注修改',(125,1395),340,28,RED,True)
note(d,1330,1662,400,240,'阅读提示','实线：业务主流程。\n虚线：相关检查支路。\n菜单可见性与自动处理取决于权限和配置。')
d.line((65,2075,1735,2075),fill='#D5E2E3',width=2)
wrapped(d,'上传成功 → 文件已到达；标注完成 → 说明已补齐；导出完成 → 训练材料已准备好。',(70,2104),1660,27,INK)
im.save(ASSETS/'01_数采到训练_完整流程.png')

# Diagram 2: proposed AI-assisted action segmentation, distinct from current capabilities.
im=Image.new('RGB',(1800,1770),BG)
d=ImageDraw.Draw(im)
header(d,'AI 如何帮助划分一段长录制','建议研究流程  ·  拟议方案，尚非平台完整现状',1800)
steps=[
    ('先定义规范','明确子任务名称、粒度，以及起止位置的判定规则。'),
    ('统一时间轴','让多路画面、关节状态和夹爪信号对应同一时刻。'),
    ('粗找候选边界','结合画面变化、夹爪开合和运动变化，找出可能的切换点。'),
    ('AI 识别步骤','按窗口阅读多视角画面，结合任务说明生成名称与初步范围。'),
    ('细看起止位置','回看候选边界附近的密集帧与动作数据，修正边界。'),
    ('人工确认草稿','在编辑器中接受、调整或驳回，然后进入审核与验收。'),
]
x,w,h,top,step=90,1120,122,235,164
for i,(title,body) in enumerate(steps):
    y=top+i*step
    d.rounded_rectangle((x,y,x+w,y+h),radius=20,fill='white',outline='#C9DBDC',width=2)
    d.text((x+25,y+23),f'{i+1:02d}',font=font(33,True),fill=TEAL)
    d.text((x+110,y+18),title,font=font(32,True),fill=INK)
    wrapped(d,body,(x+110,y+70),w-140,26)
    if i<5: arrow(d,[(x+w/2,y+h+4),(x+w/2,y+step-7)])
note(d,1280,235,430,278,'输入','长录制的多路画面\n机器人状态与动作\n主任务说明\n公司子任务标注规范')
note(d,1280,575,430,290,'要考虑的情况','双臂可以同时操作。\n等待可能是必要步骤。\n失败后的恢复可能有价值。\n不确定结果交给人工确认。',AMBER)
note(d,1280,925,430,260,'输出','步骤名称＋起止时间\n对应源帧号\n证据视角与待确认项\n保存为可修改的标注草稿',TEAL)
d.text((90,1260),'示例：给同一段录像加上动作目录',font=font(34,True),fill=INK)
wrapped(d,'以下时间与步骤仅用于说明，不是公司的正式标注标准。',(90,1316),1620,25)
segments=[('取杯','00:12—00:28','#C9E8E2',220),('放到出料位','00:28—00:43','#C7DFF2',330),('加料','00:43—01:35','#E8DEC8',530),('封口','01:35—02:10','#DCD7EF',500)]
pos=90
for name,times,color,width in segments:
    d.rounded_rectangle((pos,1382,pos+width-8,1516),radius=16,fill=color)
    d.text((pos+20,1404),name,font=font(30,True),fill=INK)
    d.text((pos+20,1463),times,font=font(22),fill=MUTED)
    pos+=width
wrapped(d,'原始录制保留完整；先确认逻辑片段，再由导出流程组织训练记录。',(90,1570),1620,29,INK,True)
wrapped(d,'评估重点：是否漏段、是否多切、边界偏差多大，以及人工修改时间是否减少。',(90,1633),1620,27)
im.save(ASSETS/'02_AI动作划分_建议流程.png')
print('Created two PNG diagrams.')
