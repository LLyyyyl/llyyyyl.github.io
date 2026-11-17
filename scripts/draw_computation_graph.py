"""Render the shared-parameter computation graph (requires matplotlib)."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from fontTools.ttLib import TTCollection
from tempfile import TemporaryDirectory
from matplotlib.patches import Circle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'static/images/notes'
# Match the article’s PingFang SC face; keep mathematical symbols in STIX.
font_dir = TemporaryDirectory(prefix='blog-figure-font-')
font_sources = list(Path('/System/Library/AssetsV2').rglob('PingFang.ttc'))
if font_sources:
 for font in TTCollection(font_sources[0]).fonts:
  if font['name'].getDebugName(1) == 'PingFang SC' and font['name'].getDebugName(2) == 'Regular':
   font_path = Path(font_dir.name) / 'PingFangSC.ttf'
   font.save(font_path)
   font_manager.fontManager.addfont(str(font_path))
   break
plt.rcParams.update({'font.family': ['PingFang SC', 'sans-serif'], 'mathtext.fontset': 'stix', 'font.size': 11, 'svg.fonttype': 'path'})
fig, axes = plt.subplots(1, 2, figsize=(12, 7.8), facecolor='white')
fig.subplots_adjust(left=.025, right=.975, top=.84, bottom=.16, wspace=.16)
pos={'x':(.8,5.3),'w':(2.5,5.3),'b':(4.2,5.3),'u':(1.25,3.8),'v':(3.75,3.8),'s':(1.25,2.3),'t':(3.75,2.3),'L':(2.5,.8)}
values={'x':3,'w':2,'b':1,'u':6,'v':3,'s':36,'t':9,'L':45}
edges=[('x','u'),('w','u'),('w','v'),('b','v'),('u','s'),('v','t'),('s','L'),('t','L')]
blue='#52748b'; orange='#a46b51'; gray='#9aa5a2'; ink='#27313a'
# Two muted branch hues, with warm neutral shared and auxiliary nodes.
node_colors = {name: ('#f3f0e9', '#787872') for name in pos}
for name in ('u', 's'):
 node_colors[name] = ('#dce8ef', blue)
for name in ('v', 't'):
 node_colors[name] = ('#efdfd5', orange)

def edge(ax,a,b,color,alpha=1,lw=1.6):
 ax.add_patch(FancyArrowPatch(pos[a],pos[b],arrowstyle='-|>',mutation_scale=11,shrinkA=23,shrinkB=23,lw=lw,color=color,alpha=alpha,zorder=1))
def label(ax,x,y,text,color=ink):
 ax.text(x,y,text,ha='center',va='center',color=color,fontsize=12,bbox=dict(facecolor='white',edgecolor='none',pad=2),zorder=4)
for i,ax in enumerate(axes):
 ax.set(xlim=(0,5),ylim=(.1,5.9),aspect='equal');ax.axis('off')
 for a,b in edges:
  if i==0 or a in ('x','b'): edge(ax,a,b,gray,alpha=.5 if i else 1,lw=1.1)
 for n,(x,y) in pos.items():
  ax.add_patch(Circle((x,y),.28,facecolor=node_colors[n][0],edgecolor=node_colors[n][1],lw=1.4,zorder=3))
  ax.text(x,y,f'${n}$',ha='center',va='center',fontsize=19,color=ink,zorder=4)
 if i==0:
  for n in ['x','w','b']:ax.text(pos[n][0],5.77,f'${n}={values[n]}$',ha='center',fontsize=13)
  label(ax,.61,3.8,r'$u=wx$');label(ax,4.43,3.8,r'$v=w+b$')
  label(ax,.61,2.3,r'$s=u^2$');label(ax,4.43,2.3,r'$t=v^2$')
  for n in ['u','v','s','t']:ax.text(pos[n][0]+.38,pos[n][1]-.38,f'$={values[n]}$',fontsize=12,color='#66737d')
  label(ax,2.5,.23,r'$L=s+t=45$')
 else:
  for a,b in [('L','s'),('s','u'),('u','w')]:edge(ax,a,b,blue,lw=2)
  for a,b in [('L','t'),('t','v'),('v','w')]:edge(ax,a,b,orange,lw=2)
  label(ax,1.37,1.21,r'$1\times1=1$',blue);label(ax,3.65,1.21,r'$1\times1=1$',orange)
  label(ax,1.1,3.04,r'$1\times2u=12$',blue);label(ax,3.9,3.04,r'$1\times2v=6$',orange)
  label(ax,2.15,4.38,r'$12\times x=36$',blue);label(ax,3.02,4.72,r'$6\times1=6$',orange)
  label(ax,2.5,5.80,r'$\frac{\partial L}{\partial w}=36+6=42$')
  label(ax,2.5,.21,r'$\frac{\partial L}{\partial L}=1$')
 axes[i].set_title(['（a）前向：建立计算依赖','（b）反向：汇总梯度贡献'][i],fontsize=13,fontweight='normal',pad=20,color=ink)
fig.text(.5,.975,r'$L=(wx)^2+(w+b)^2$',ha='center',va='top',fontsize=22,color=ink)
fig.text(.5,.11,r'$\frac{\partial L}{\partial w}=\frac{\partial L}{\partial s}\frac{\partial s}{\partial u}\frac{\partial u}{\partial w}+\frac{\partial L}{\partial t}\frac{\partial t}{\partial v}\frac{\partial v}{\partial w}=1\cdot12\cdot3+1\cdot6\cdot1=42$',ha='center',fontsize=18,color=ink)
fig.text(.5,.035,'节点表示张量，连线表示依赖；彩色箭头表示传向共享参数的梯度贡献。',ha='center',fontsize=13,color='#65717b')
OUT.mkdir(parents=True,exist_ok=True)
for ext in ['png','svg']:fig.savefig(OUT/f'computation-graph.{ext}',dpi=200,facecolor='white')
# Normalize generated SVG whitespace for clean version-control diffs.
svg_path = OUT / 'computation-graph.svg'
svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines()) + '\n')
plt.close(fig)
