---
title: "模型训练基础：PyTorch 训练循环"
date: '2026-11-16'
draft: false
description: "以一个最小 MLP 为例，梳理 PyTorch 训练迭代中计算图的建立、梯度的生成与清理，以及参数和优化器状态的更新。"
tagsInHeader: true
math: true
trainingNote: true
categories:
  - "AI Infrastructure"
tags:
  - "PyTorch"
  - "Training"
---

## 一、基本概念 {#基本概念}

### 1. 训练中的计数单位 {#一训练中的计数单位}

- `batch`：一次送入模型的一组样本，样本数量称为 batch size。
- `iteration`：训练循环处理一个 batch 的过程，包括前向计算、损失计算和反向传播。
- `optimizer step`：优化器根据当前累积的梯度执行一次参数更新。
- `epoch`：完整遍历一次训练集，通常包含多个 iteration。训练多个 epoch 时，会重复使用训练数据，常在每个 epoch 开始时重新打乱样本顺序。

### 2. 批量大小与梯度累积 {#批量大小与梯度累积}

- `MBS`（Micro Batch Size）：每个数据并行副本一次处理的样本数。当每张 GPU 各自保存一份完整模型时，MBS 就是每张 GPU 一次接收的 batch 大小，各 GPU 分别用各自的样本计算梯度。
- `梯度累积`：连续处理多个 micro-batch，将梯度累积后再执行一次参数更新。组内不清理梯度，也不更新参数。
- `GBS`（Global Batch Size）：一次参数更新使用的总样本数，包含所有数据并行副本及各次梯度累积处理的样本。

记梯度累积步数为 \(K\)，数据并行副本数为 \(D\)，各副本的 MBS 相同时，\(\mathrm{GBS}=\mathrm{MBS}\times K\times D\)。各副本每组均累积 \(K\) 次，且各 micro-batch 大小相同时，通常将每次的平均损失除以 \(K\)，使累积后的梯度对应整组样本的平均损失。

{{< note-example title="数据并行与梯度累积" >}}
例如，4 张 GPU 各保存一份完整模型，每张 GPU 每批处理 8 个不同的样本，即 \(\mathrm{MBS}=8\)。

每批更新一次：\(K=1\)，\(\mathrm{GBS}=8\times1\times4=32\)。<br>
累积 4 批后更新：\(K=4\)，\(\mathrm{GBS}=8\times4\times4=128\)。
{{< /note-example >}}

## 二、最小训练循环 {#1-基本训练流程}

以两层 MLP 在合成二分类数据上的训练为例，采用交叉熵损失和 Adam 优化器。

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

# 32 个样本，每个样本有 4 个特征；标签为 0 或 1。
x = torch.randn(32, 4)
y = (x[:, 0] + x[:, 1] > 0).long()
loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True)

# 定义模型
model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 2),
)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()  # 切换到训练模式
for epoch in range(10):
    total_loss = 0.0
    for inputs, targets in loader:
        optimizer.zero_grad(set_to_none=True)  # zero_grad：清理上一轮梯度
        logits = model(inputs)                 # forward：计算预测分数
        loss = criterion(logits, targets)      # loss：计算当前 batch 的损失
        loss.backward()                        # backward：计算参数梯度
        optimizer.step()                       # step：更新参数

        total_loss += loss.item() * inputs.size(0)

    mean_loss = total_loss / len(loader.dataset)
    print(f"epoch={epoch + 1:02d}, loss={mean_loss:.4f}")
```

`model.train()` 用于切换{{< tooltip term="训练模式" >}}
模式切换会改变模型中 Dropout、BatchNorm 等层的行为，前提是模型包含这些层。

`model.train()` 表示**训练模式**：<br>
训练模式下，Dropout 启用随机置零，以减轻过拟合；默认配置的 BatchNorm 使用当前 batch 的均值和方差，并更新保存的运行统计量。

`model.eval()` 表示**评估模式**：<br>
评估模式用于验证或预测，Dropout 停止随机置零；默认配置的 BatchNorm 使用保存的运行统计量，不再更新它们。

{{< /tooltip >}}，本身不会执行训练或更新参数。

<figure class="training-diagram" aria-label="一次训练迭代的五个步骤">
<ol class="training-flow">
<li><code>zero_grad()</code><span>清理旧梯度</span></li>
<li><code>forward()</code><span>计算预测</span></li>
<li><code>criterion()</code><span>计算损失</span></li>
<li><code>backward()</code><span>累积参数梯度</span></li>
<li><code>step()</code><span>更新参数</span></li>
</ol>
</figure>

## 三、前向传播：从输入到损失 {#2-前向传播与损失计算}

### 1. 输入到预测 {#一输入到预测}

前向传播通过 `model(inputs)` 执行，按照模型定义的计算过程，将输入转换为输出。输出的含义取决于任务与模型设计：分类模型可以输出类别分数，回归模型可以输出连续数值，特征提取模型则可以输出向量表示。<strong>前向传播使用当前参数进行计算，不执行优化器的参数更新。</strong>

在上述二分类训练示例中，输入依次经过线性层、ReLU 和第二个线性层，得到两个类别对应的原始分数 `logits`。随后，损失函数根据 `logits` 与真实标签 `targets` 计算损失。

### 2. 损失计算 {#二损失计算}

<strong>损失函数</strong><strong class="note-key-point">将模型输出与目标之间的差异量化为损失</strong><strong>，定义训练所要最小化的目标。</strong>

在上述训练示例中，`criterion = nn.CrossEntropyLoss()` 定义了交叉熵损失函数。执行 `loss = criterion(logits, targets)` 时，将模型输出与类别索引形式的标签传入，计算当前 batch 的损失。损失计算本身不更新参数。

<p>（<a href="#常见损失函数">常见损失函数</a>）</p>

### 3. 为反向传播记录计算图 {#三为反向传播记录计算图}

执行 `model(inputs)` 时，PyTorch 不只是在计算预测值。在默认开启梯度记录的情况下，autograd 还会记录参与求导的运算及其依赖关系，构建用于反向传播的{{< tooltip term="计算图" >}}
**计算图**记录了“结果经过哪些运算得到，以及这些运算依赖哪些数据”。例如，`y = w * x + b`，再由 `y` 计算 `loss`，就形成了从参数到损失的依赖关系。

它的作用是**支持自动求导**：调用 `loss.backward()` 时，PyTorch 沿这些关系反向应用链式法则，计算损失对参数的梯度。<br>随后由 `optimizer.step()` 使用梯度更新参数。
{{< /tooltip >}}。随后计算 `loss` 的运算也会接到图中，因此可以从损失出发，沿依赖关系找到影响它的模型参数。

默认执行方式为{{< tooltip term="eager" >}}
<strong>Eager（即时执行）</strong>：Python 执行到一个张量运算，就发起该运算，便于查看中间结果和调试。这里的“即时”不代表每一步都等待 GPU 运算完成。

<strong>Compile（编译执行）</strong>：通过 `torch.compile` 捕获可编译的运算片段，编译优化后执行，并在后续调用中尽可能复用编译结果。首次编译有额外开销，后续执行可能更快。

两者区别在于运算的执行方式。<strong>Eager 同样会建立用于自动求导的计算图</strong>，是否记录梯度由梯度模式和张量的 `requires_grad` 等条件决定。
{{< /tooltip >}}。autograd 在记录运算依赖的同时，也会按反向求导的需要保存部分中间结果。

## 四、反向传播：从损失到参数梯度 {#3-反向传播与梯度}

前向传播得到模型的预测输出 \(A\)。损失函数将它与真实目标 \(B\) 比较，计算出损失 \(L\)。训练的目标是反复调整模型参数，让训练数据上的损失整体降低。<strong>每一步更新时，如何判断哪个方向能降低损失？</strong>

### 1. 梯度下降与反向传播 {#一backward-的计算过程}

梯度下降的做法是沿负梯度方向更新参数。把损失写成参数 \(\theta\) 的函数 \(L(\theta)\)。在当前参数处对损失求梯度，记为 \(g=\nabla_\theta L(\theta)\)。**梯度的每个元素都是损失对相应参数的偏导数。**

如果把参数从 \(\theta\) 小幅调整到 \(\theta+\Delta\theta\)，就可以用一阶近似将梯度与损失的变化联系起来。

\[
\Delta L=L(\theta+\Delta\theta)-L(\theta)\approx g^\top\Delta\theta.
\]

根据当前参数 \(\theta\) 可以计算出梯度 \(g\)，再利用一阶近似估计参数变化对损失的影响。要使损失下降（\(\Delta L\lt 0\)），梯度下降选择沿负梯度方向做小幅更新。设学习率为 \(\eta>0\)，参数更新公式为

\[
\theta_{\mathrm{new}}=\theta+\Delta\theta=\theta-\eta g.
\]

在训练循环中，**`loss.backward()` 负责计算梯度，`optimizer.step()` 再按优化器的规则更新参数。**

参数通常经过多层运算才影响损失。以简化的标量依赖链 \(\theta\to h\to L\) 为例，反向传播沿这条路径应用链式法则。

\[
\frac{\partial L}{\partial\theta}
=\frac{\partial L}{\partial h}
\frac{\partial h}{\partial\theta}.
\]

沿这条路径，将损失对 \(h\) 的梯度与 \(h\) 对参数 \(\theta\) 的导数相乘，即得到损失对该参数的梯度。

反向计算从损失端沿依赖关系向前面的运算推进：

1. 每个反向节点接收损失对其输出的梯度。
2. 结合该运算的求导规则和必要的前向数据，计算损失对其输入的梯度，并继续传递。
3. 同一个张量通过多条路径影响损失时，汇总各条路径的梯度贡献。
4. 到达需要求梯度的叶子参数后，将结果累积到对应参数的 `.grad` 中，供后续 `optimizer.step()` 使用。

对于向量、矩阵等张量运算，autograd 执行相应的链式求导，不需要显式构造完整的雅可比矩阵。反向过程中会计算中间张量的梯度，但默认只把需要求梯度的叶子张量的结果保存在 `.grad` 中。

{{< note-example title="共享参数的计算图与梯度汇总" >}}
以标量目标 \(L=(wx)^2+(w+b)^2\) 为例，取 \(x=3\)、\(w=2\)、\(b=1\)，观察同一个参数 \(w\) 通过两条路径影响损失的情况。

![共享参数计算图：左图展示前向依赖，右图展示两条反向路径在参数 w 处汇总梯度](/images/notes/computation-graph.png?v=two-branches-3)

**前向构图**：先计算 \(u=wx\)、\(v=w+b\)，再计算 \(s=u^2\)、\(t=v^2\)，最后相加得到 \(L=45\)。每执行一个运算，就建立其输出与输入之间的依赖；\(u\) 和 \(v\) 使用的是同一个 \(w\) 节点。

**反向求导**：从 \(L\) 出发，沿依赖关系反向传递梯度，每一步乘以当前运算的局部导数。蓝色路径贡献 \(1\times2u\times x=36\)，橙色路径贡献 \(1\times2v\times1=6\)。

<strong>同一个参数通过多条路径影响损失时，各条路径的梯度贡献需要相加。</strong> 因此 \(\frac{\partial L}{\partial w}=36+6=42\)。
{{< /note-example >}}

### 2. 求导属性 {#二求导属性}

<div class="autograd-table-wrap" tabindex="0" role="region" aria-label="求导属性对照表">
<table class="autograd-table">
<thead><tr><th scope="col">属性</th><th scope="col">含义</th><th scope="col">张量中的表现</th></tr></thead>
<tbody>
<tr>
<th scope="row"><code>requires_grad</code></th>
<td>是否需要跟踪该张量相关的梯度计算</td>
<td><span class="tensor-example">模型参数 <code>True</code></span><span class="tensor-example">普通输入、整数标签 <code>False</code></span></td>
</tr>
<tr>
<th scope="row"><code>grad_fn</code></th>
<td>记录该张量对应的反向求导操作，用于将梯度传向上游</td>
<td><span class="tensor-example"><code>logits</code>、<code>loss</code> 有反向节点</span><span class="tensor-example">模型参数通常为 <code>None</code></span></td>
</tr>
<tr>
<th scope="row"><code>is_leaf</code></th>
<td>是否为叶子张量</td>
<td><span class="tensor-example">直接创建的模型参数为叶子</span><span class="tensor-example"><code>logits</code>、<code>loss</code> 为非叶子</span></td>
</tr>
</tbody>
</table>
</div>

模型参数通常由 `nn.Parameter` 创建，默认 `requires_grad=True`。虽然 `inputs` 的 `requires_grad=False`，不需要计算损失对输入的梯度，但模型输出还依赖权重和偏置，训练时需要计算损失对这些参数的梯度。因此，在默认梯度模式下，autograd 仍会记录相关运算，保留从模型参数到 `logits`、再到 `loss` 的计算依赖。

`logits` 和 `loss` 由已记录的运算产生，其 `grad_fn` 指向这些运算对应的反向求导操作，用于计算并向前面的张量传递梯度。模型参数通常是直接创建的叶子张量，而非由计算图中的运算产生，因此其 `grad_fn` 为 `None`。

<strong><code>grad_fn=None</code> 不表示不能获得梯度。</strong> 只要损失依赖该参数且参数需要求梯度，反向传播仍会计算损失对它的梯度。

对于需要求梯度的张量，直接创建、不是由已记录运算产生的张量是叶子张量。例如，模型权重是叶子，而 `logits` 和 `loss` 是非叶子。无需求梯度的张量也被视为叶子，因此<strong>叶子张量不一定是可训练参数</strong>。

参与求导的叶子参数默认将梯度累积到 `.grad` 中。中间非叶子张量的梯度也参与反向计算，但默认不保存在它们自己的 `.grad` 中。

### 3. 梯度存储与清理 {#三梯度存储与清理}

`.grad` 存放累积得到的梯度数值，与参数张量的形状相同，每个梯度元素对应一个参数元素。

<span class="note-inline-call">optimizer.zero_grad({{< tooltip term="set_to_none=True" >}}
<strong>True：将梯度设为 <code>None</code></strong><br>参数的 `.grad` 不再保存旧梯度张量，下一次反向传播时再写入新的梯度。

<strong>False：将已有梯度置零</strong><br>保留已有梯度张量，将其中的元素清零。原本为 `None` 的 `.grad` 仍保持 `None`。

<strong>更新时的区别</strong><br>对 SGD、Adam 等优化器，<strong><code>.grad=None</code> 会跳过该参数的更新</strong>。<br><strong>零梯度仍会进入更新逻辑</strong>，参数可能因动量或权重衰减而变化。

两种方式都<strong>只清理梯度，不改变参数值，也不清空优化器的历史状态</strong>。
{{< /tooltip >}})</span> 将优化器管理的参数的 `.grad` 设为 `None`，清理上一轮的梯度。执行 `loss.backward()` 后，本轮参与求导的参数会在 `.grad` 中累积得到新的梯度，未获得梯度的参数则仍为 `None`。

默认情况下，<strong>新梯度会累加到已有的 <code>.grad</code> 上，而不是自动覆盖</strong>。

## 五、参数更新：从梯度到新参数 {#4-参数更新与梯度清理}

反向传播把梯度写入 `.grad` 后，优化器才能据此更新参数。<strong><code>optimizer.step()</code> 负责更新，<code>optimizer.zero_grad()</code> 负责清理梯度。</strong>

### 1. optimizer.step：使用梯度更新参数 {#一optimizerstep使用梯度更新参数}

以 SGD 为例，参数沿负梯度方向更新，学习率控制更新幅度：

\[
\theta_{t+1}=\theta_t-\eta g_t,
\qquad g_t=\nabla_\theta L(\theta_t).
\]

调用 `optimizer.step()` 时，SGD 根据 `.grad` 中的梯度和学习率，按上述规则更新参数。

在上述训练循环中，Adam 的 `optimizer.step()` 直接读取现有梯度并更新参数和 Adam 状态，不重新执行前向、反向，<strong>也不自动清空 <code>.grad</code></strong>。更新后的参数要到下一次前向时才会产生新的预测与损失。

<p>（<a href="#常见优化器">常见优化器</a>）</p>

### 2. zero_grad：为下一轮反向传播做准备 {#二zero_grad为下一轮反向传播做准备}

未使用梯度累积时，在每轮前向前调用 `optimizer.zero_grad(set_to_none=True)`，使随后 `backward()` 得到的是当前 batch 的梯度。也可以在上一次 `step()` 后清理，但必须保证第一次反向前没有残留梯度，并且<strong>不能在 <code>backward()</code> 与消费这些梯度的 <code>step()</code> 之间清理</strong>。

## 六、一次迭代中的状态生命周期 {#6-一次迭代中的数据生命周期}

默认 eager 执行、每个 batch 更新一次且 `set_to_none=True` 时，各类状态的生命周期如下。

<div class="autograd-table-wrap" tabindex="0" role="region" aria-label="训练状态生命周期对照表">
<table class="autograd-table lifecycle-table">
<thead><tr><th scope="col">阶段</th><th scope="col">模型参数</th><th scope="col">参数梯度 <code>.grad</code></th><th scope="col">计算图与中间结果</th><th scope="col">优化器状态</th></tr></thead>
<tbody>
<tr><th scope="row"><code>forward()</code> 前</th><td>保留当前参数</td><td><code>None</code></td><td>本轮图尚未建立</td><td>保留已有状态（若有）</td></tr>
<tr><th scope="row">损失计算后</th><td>数值不变</td><td><code>None</code></td><td>建立本轮图<br>保存 <code>backward()</code> 需要的数据</td><td>不变</td></tr>
<tr><th scope="row"><code>backward()</code> 后</th><td>数值不变</td><td>得到本轮梯度</td><td>释放为反向求导保存的数据</td><td>不变</td></tr>
<tr><th scope="row"><code>step()</code> 后</th><td>按规则更新</td><td>保留刚才的梯度</td><td>更新不依赖重新执行本轮图</td><td>初始化或更新状态（若有）</td></tr>
<tr><th scope="row">下一轮清理梯度后</th><td>保留更新后参数</td><td>再次设为 <code>None</code></td><td>下一次 <code>forward()</code> 建立新的图</td><td>保留更新后的状态（若有）</td></tr>
</tbody>
</table>
</div>

跨迭代延续的是<strong class="note-key-point">模型参数和优化器状态</strong>。梯度在下一轮开始前清理，中间激活和计算图则围绕本轮 `forward()`、`backward()` 产生与使用。

**在开启梯度记录、运算涉及需要求梯度的张量时，eager 模式会随每轮前向重新建立计算图。** 模型结构描述层与运算的组织方式，本轮计算图则记录这一次实际执行的运算及依赖，并保存反向求导所需的数据。

例如，每轮都计算 \(h=Wx\)，运算形式可以相同，但反向求导需要使用本轮前向的数据。eager 模式按实际执行的运算动态记录依赖，因此可以支持随输入变化的条件分支。这里的重新构图指 eager 下的自动求导记录，不等同于 `torch.compile` 对编译结果的复用。

**重新构图不等于重新创建模型参数。** 参数跨迭代保留，其数值由优化器更新。`retain_graph=True` 用于保留同一次前向的计算图及反向所需的数据，以便再次反向求导，并不会让下一轮前向自动复用旧图。

## 七、损失函数与优化器 {#补充说明}

<span id="损失函数与优化器"></span>

<strong class="note-key-point">损失函数定义优化目标，优化器根据梯度决定参数的更新方式。</strong> <br>
两者通过反向传播连接：损失函数计算 `loss`，`backward()` 将参数梯度累积到 `.grad`，`optimizer.step()` 再据此更新参数。

损失函数与优化器之间不存在固定的对应关系。交叉熵损失可结合 SGD、Adam 或 AdamW 进行优化，同一优化器也可用于不同损失函数所定义的优化目标。

### 1. 常见损失函数 {#常见损失函数}

损失函数用于量化模型预测与目标之间的差异。不同任务的预测形式不同，对应的损失函数也有所区别。<br><strong class="note-key-point">分类任务</strong>关注预测的类别概率与目标标签之间的差异，常用交叉熵或二元交叉熵。<br><strong class="note-key-point">回归任务</strong>关注预测数值与目标值之间的偏差，常用 MSE、L1 或 Smooth L1。

#### 1.1 分类损失 {#1-分类损失}

交叉熵用于互斥类别的预测，二元交叉熵用于二元判断或多个分别判断的标签。

##### 1.1.1 交叉熵：`CrossEntropyLoss` {#1交叉熵crossentropyloss}

常用于单标签分类任务，每个样本对应一个目标类别，例如将一张图片归为“猫”“狗”或“鸟”。

设共有 \(C\) 个类别，\(z_c\) 为第 \(c\) 类的 logit，真实类别编号为 \(y\)。从数学定义看，softmax 将各类别的分数转成概率，再取真实类别概率的负对数：

\[
\begin{gathered}
p_c = \frac{e^{z_c}}{\sum_{j=1}^{C} e^{z_j}} \\[6pt]
\ell = -\log p_y
\end{gathered}
\]

上述公式展示数学关系，`CrossEntropyLoss` 直接接收 logits，无需在调用前手动执行 softmax。

真实类别的预测概率越低，损失越大；概率越高，损失越小。

{{< note-example title="交叉熵与真实类别概率" >}}
以真实类别为“猫”的图片为例，取模型为“猫”分配的概率计算损失：

模型预测为“猫”的概率为 \(0.1\) 时，损失为 \(\ell=-\log 0.1\approx2.30\)。

模型预测为“猫”的概率提高到 \(0.9\) 时，损失降为 \(\ell=-\log 0.9\approx0.11\)。

真实类别的预测概率提高，交叉熵损失随之减小。
{{< /note-example >}}

**🤖 LLM 应用：token 级交叉熵**

自回归语言模型在每个位置根据前文预测下一个 token，相当于在整个词表上做多分类。<strong>预训练通常使用这一目标，监督微调（SFT）也通常沿用交叉熵</strong>，只是训练数据和参与损失计算的位置不同。

\[
L_{\mathrm{CE}}=-\frac{1}{\sum_t m_t}\sum_t m_t\log p_\theta(x_t\mid x_{\lt t})
\]

公式按“逐位置计算、筛选、取平均”得到整条序列的损失：

- **单个位置**：\(x_t\) 是第 \(t\) 个位置的真实 token，\(x_{\lt t}\) 是它之前的 token 序列。\(p_\theta(x_t\mid x_{\lt t})\) 表示参数为 \(\theta\) 的模型根据前文，为真实 token 分配的概率。该位置的损失为 \(-\log p_\theta(x_t\mid x_{\lt t})\)，概率越高，损失越小。
- **掩码筛选**：\(m_t=1\) 表示该位置参与损失计算，\(m_t=0\) 表示忽略该位置。因此，分子只累加被选中位置的负对数概率。
- **取平均**：\(\sum_t m_t\) 是参与计算的位置数。将累计损失除以这个数量，得到每个有效 token 的平均损失 \(L_{\mathrm{CE}}\)。

预训练通常计算有效文本位置的损失，而 SFT 可以只计算回答部分或 assistant 消息的损失，提示词、padding 等位置通过标签掩码排除。训练时使用真实前文，预测与目标需错开一个 token 对齐。

##### 1.1.2 二元交叉熵：`BCEWithLogitsLoss` {#2二元交叉熵bcewithlogitsloss}

常用于二分类或多标签分类任务。二分类判断样本是否属于某个类别，例如判断一张图片中是否有猫；多标签分类则对多个标签分别判断，例如一张图片可以同时具有“猫”和“室内”两个标签。

对一个二元标签，\(z\) 表示 logit，\(y\) 表示目标值。从数学定义看，先通过 sigmoid 将分数转成概率，再计算二元交叉熵：

\[
\begin{gathered}
p = \sigma(z) = \frac{1}{1+e^{-z}} \\[6pt]
\ell = -\bigl[y\log p + (1-y)\log(1-p)\bigr]
\end{gathered}
\]

`BCEWithLogitsLoss` 已将 sigmoid 与二元交叉熵合并，直接接收 logits。

当 \(y=1\) 时，希望 \(p\) 接近 1；当 \(y=0\) 时，希望 \(p\) 接近 0。多标签分类中，各标签分别计算损失，**预测概率不要求加起来等于 1**。

#### 1.2 回归损失 {#2-回归损失}

MSE、L1、Huber 和 Smooth L1 的主要区别在于如何惩罚预测误差。以下用 \(\hat y\) 表示预测值，\(y\) 表示目标值。

##### 1.2.1 均方误差：`MSELoss` {#1均方误差mseloss}

常用于预测温度、价格等连续数值的回归任务。

单个元素的损失是预测误差的平方：

\[
\ell = (\hat y-y)^2
\]

由于损失随预测误差的平方增长，**较大的误差会受到更强的惩罚**。例如，预测偏差从 1 增加到 3 时，损失从 1 增加到 9。因此，MSE 对偏离目标较远的预测更敏感，也更容易受离群值影响。

##### 1.2.2 绝对误差：`L1Loss` {#2绝对误差l1loss}

常用于需要减弱离群值影响的回归任务，例如房价预测中，减弱少量异常样本造成的大误差对训练的影响。

单个元素的损失是预测误差的绝对值：

\[
\ell = |\hat y-y|
\]

由于损失随误差的绝对值线性增长，**较大误差不会像 MSE 中那样受到平方惩罚**。例如，预测偏差从 1 增加到 3 时，L1 损失也从 1 增加到 3，而 MSE 损失会从 1 增加到 9。因此，相比 MSE，L1 对离群值通常更不敏感。

误差不为零时，损失对预测值的梯度为误差的符号（\(\pm1\)），其大小不会随误差减小而减小。误差为零时，左右两侧斜率不同，形成不可导的折点；实现中会约定该点的梯度取值。

##### 1.2.3 Huber 损失：`HuberLoss` {#3huber-损失huberloss}

常用于需要兼顾零点附近的平滑性与对离群值稳健性的回归任务，例如传感器数值预测中，既拟合正常变化，又减弱偶发异常读数的影响。

记误差 \(e=\hat y-y\)，人为设定阈值 \(\delta>0\)，用于划分二次惩罚与线性惩罚的区间。\(\delta\) 是超参数，通常根据目标值的尺度和验证效果选择。损失定义为：

\[
\ell_\delta(e)=
\begin{cases}
\dfrac{1}{2}e^2, & |e|\lt\delta, \\
\delta\left(|e|-\dfrac{\delta}{2}\right), & |e|\ge\delta.
\end{cases}
\]

当误差的绝对值小于 \(\delta\) 时，损失随误差的平方增长，损失对预测值的梯度也随误差趋近零而减小；当误差的绝对值达到或超过 \(\delta\) 时，损失转为线性增长，梯度的绝对值保持为 \(\delta\)。这样既保留了零点附近的平滑性，也限制了大误差对梯度的影响。

##### 1.2.4 平滑 L1：`SmoothL1Loss` {#4平滑-l1smoothl1loss}

常用于目标检测中的边界框回归等任务，在减弱大误差影响的同时，保持零点附近的平滑性。例如，预测图片中车辆的位置和边界框大小。

记误差 \(e=\hat y-y\)，人为设定阈值 \(\beta>0\)，用于划分二次惩罚与线性惩罚的区间。与 Huber 中的 \(\delta\) 类似，\(\beta\) 也是超参数，通常根据目标值的尺度和验证效果选择。损失定义为：

\[
\ell =
\begin{cases}
\dfrac{e^2}{2\beta}, & |e|\lt \beta, \\
|e|-\dfrac{\beta}{2}, & |e|\ge\beta.
\end{cases}
\]

当误差的绝对值小于 \(\beta\) 时，损失采用二次形式，损失对预测值的梯度为 \(e/\beta\)，随误差趋近零而减小；当误差的绝对值达到或超过 \(\beta\) 时，损失转为线性增长，梯度的绝对值保持为 1。这样既使零点附近的变化更平滑，也避免大误差导致梯度持续增大。

当 \(\beta=0\) 时，不再使用二次区间，损失退化为 L1。

Smooth L1 与 Huber 存在缩放关系。当两者阈值相同，即 \(\delta=\beta>0\) 时：

\[
\mathrm{SmoothL1}_\beta(e)=\frac{\mathrm{Huber}_\beta(e)}{\beta}.
\]

<strong>阈值都为 1 时，两者完全相同。</strong> 其他阈值下，两者的损失及梯度相差一个缩放系数：Smooth L1 在线性区域的梯度大小为 1，Huber 则为 \(\delta\)。

### 2. 常见优化器 {#常见优化器}

各优化器接收的都是参数梯度，区别在于如何利用历史信息、形成更新量。

<div class="autograd-table-wrap" tabindex="0" role="region" aria-label="常见优化器对照表">
<table class="autograd-table optimizer-table">
<thead><tr><th scope="col">优化器</th><th scope="col">更新方式</th><th scope="col">主要特点</th></tr></thead>
<tbody>
<tr><th scope="row">SGD</th><td>沿当前梯度的反方向更新</td><td>加入动量后结合历史方向</td></tr>
<tr><th scope="row">Adam</th><td>结合梯度历史，逐元素调整更新幅度</td><td>维护一阶矩与二阶矩估计</td></tr>
<tr><th scope="row">AdamW</th><td>Adam 更新与权重衰减分别执行</td><td>衰减项不进入矩估计</td></tr>
<tr><th scope="row">Muon</th><td>对动量矩阵进行近似正交化</td><td>利用矩阵结构调整更新方向</td></tr>
</tbody>
</table>
</div>

#### 2.1 SGD（Stochastic Gradient Descent，随机梯度下降） {#1-sgdstochastic-gradient-descent随机梯度下降}

SGD 使用抽样数据估计整体训练目标的梯度。实际训练通常采用 mini-batch SGD，即用一个 batch 的梯度更新参数，而不是每次遍历全部训练数据。

##### 2.1.1 基本更新规则 {#1基本更新规则}

设 \(\theta_t\) 为当前参数，\(B_t\) 为第 \(t\) 次更新使用的 batch，\(L_{B_t}\) 为该 batch 的平均损失，\(\eta_t\) 为学习率。不带动量和权重衰减时：

\[
g_t=\nabla_\theta L_{B_t}(\theta_t),
\qquad
\theta_{t+1}=\theta_t-\eta_t g_t.
\]

在当前参数 \(\theta_t\) 处，对 batch 损失 \(L_{B_t}\) 求导，计算得到参数梯度 \(g_t\)。随后按学习率 \(\eta_t\) 沿负梯度方向更新参数，得到 \(\theta_{t+1}\)。当前 batch 的梯度只是整体训练目标梯度的估计，因此一次更新不保证整体训练损失下降。

##### 2.1.2 动量：减轻振荡，积累一致的更新方向 {#2动量减轻振荡积累一致的更新方向}

基本 SGD 仅依据当前 batch 的梯度更新参数。由于不同 batch 的样本组成不同，计算出的梯度方向和大小可能存在差异，使参数更新方向在迭代之间发生波动。即使使用确定的梯度，在曲率差异较大的区域，更新也可能在某些方向上反复振荡，影响收敛效率。

<strong class="note-key-point">动量通过保留逐渐衰减的历史梯度，使更新方向同时考虑当前梯度和历史趋势。</strong>方向一致的梯度贡献会累积，方向相反的贡献会部分抵消，从而有助于减轻振荡，并加快沿持续下降方向的移动。

设 \(\mu\) 为动量系数，\(v_{t-1}\) 为上一轮的动量缓冲，初始历史缓冲取零。采用标准动量更新、不考虑权重衰减时：

\[
v_t=\mu v_{t-1}+g_t,
\qquad
\theta_{t+1}=\theta_t-\eta_t v_t.
\]

将历史缓冲 \(v_{t-1}\) 乘以动量系数 \(\mu\)，再与当前梯度 \(g_t\) 相加，计算得到本轮动量缓冲 \(v_t\)。随后沿 \(-v_t\) 方向更新参数。动量在每次参数更新时保留并利用历史梯度信息，以影响后续的更新方向；梯度累积则在一次参数更新前，汇总多个 micro-batch 的梯度，再执行一次更新。

##### 2.1.3 权重衰减：约束参数规模，改善泛化 {#3权重衰减约束参数规模改善泛化}

模型在降低训练损失的过程中，既可能学到具有普遍性的规律，也可能拟合训练数据中特有的噪声。因此，训练集上的表现变好，并不一定意味着模型在新数据上也表现更好。为减轻过拟合，可以引入权重衰减，对参数规模施加约束。

权重衰减在参数更新时，额外使权重向零靠近。<strong>其目的是降低过拟合风险，而不是单纯加快训练损失下降。</strong> 衰减过强也可能导致欠拟合，需要通过系数控制约束程度。

设 \(\lambda\) 为权重衰减系数。不考虑动量时，在当前梯度中加入衰减项 \(\lambda\theta_t\)，参数更新为：

\[
\tilde g_t=g_t+\lambda\theta_t,
\qquad
\theta_{t+1}=\theta_t-\eta_t\tilde g_t=(1-\eta_t\lambda)\theta_t-\eta_t g_t.
\]

\(\tilde g_t\) 为加入衰减项后的梯度，由 \(g_t+\lambda\theta_t\) 得到。使用它更新参数，相当于先将当前参数乘以 \(1-\eta_t\lambda\)，再执行梯度更新。这一衰减作用独立于损失梯度的方向。若同时启用动量，\(\tilde g_t\) 会进入动量缓冲。<strong>权重衰减作用于参数，与学习率随训练进程衰减是不同的机制。</strong>

{{< note-example title="SGD 权重衰减的更新过程" >}}
以模型 \(\hat y=wx\)、平方误差 \(L=(\hat y-y)^2\) 为例，设当前权重 \(w=2\)，输入 \(x=1\)，目标 \(y=3\)，学习率 \(\eta=0.1\)，衰减系数 \(\lambda=0.1\)。

**计算梯度**：前向得到 \(\hat y=2\)、\(L=1\)，反向计算得到 \(g_t=2(wx-y)x=-2\)。

**加入衰减**：\(\tilde g_t=g_t+\lambda w=-2+0.2=-1.8\)。

**更新参数**：\(w_{\text{new}}=w-\eta\tilde g_t=2.18\)。不使用权重衰减时，权重会更新为 \(2.2\)。

权重仍然增大，但衰减项使其增幅减小。下一轮使用更新后的权重和当前 batch，重新计算梯度 \(g_{t+1}\)。
{{< /note-example >}}

**优化器状态**：基本 SGD 无需维护逐参数的历史状态。启用动量后，需要保存 `momentum_buffer`，供后续更新使用。

#### 2.2 Adam（Adaptive Moment Estimation，自适应矩估计） {#2-adamadaptive-moment-estimation自适应矩估计}

SGD 的动量为各参数积累历史梯度，但同一参数组中的参数在更新时仍使用同一个学习率进行缩放，不会根据各参数的历史梯度尺度分别调整缩放比例。不同参数的梯度尺度可能相差较大，统一的缩放方式难以同时适应这些差异。

<strong class="note-key-point">Adam 同时记录梯度的历史方向和大小，为不同参数元素调整更新幅度。</strong>

##### 2.2.1 梯度历史：一阶矩与二阶矩 {#1梯度历史一阶矩与二阶矩}

Adam 为每个参数维护两类历史统计量：用 \(m_t\) 记录梯度的指数移动平均，用 \(v_t\) 记录梯度平方的指数移动平均，分别用于估计更新方向和调整更新幅度。此处的 \(v_t\) 表示梯度平方的统计量，与前文 SGD 的动量缓冲含义不同。

设 \(\theta_t\) 为当前参数，\(g_t\) 为当前 batch 的参数梯度。\(\beta_1\)、\(\beta_2\) 分别控制两类历史统计的保留程度，初始状态 \(m_0=v_0=0\)，更新次数从 \(t=1\) 开始：

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
\]

其中，\(g_t^2\) 表示对梯度逐元素平方。

##### 2.2.2 偏差修正：减小零初始化的影响 {#2偏差修正减小零初始化的影响}

两类统计从零开始，在训练初期会偏向零。为修正这一初始化效应，分别除以对应的累计权重：

\[
\hat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}.
\]

{{< note-example title="Adam 一阶矩的偏差修正" >}}
以一阶矩为例，假设每次梯度均为 10，\(\beta_1=0.9\)，初始 \(m_0=0\)：

**第一次更新**：\(m_1=0.9\times0+0.1\times10=1\)。真实梯度只占 0.1 的权重，其余权重来自初始零值。除以 \(1-0.9=0.1\)，得到 \(\hat m_1=10\)。

**第二次更新**：\(m_2=0.9\times1+0.1\times10=1.9\)。真实梯度的累计权重为 \(1-0.9^2=0.19\)，因此 \(\hat m_2=1.9/0.19=10\)。
{{< /note-example >}}

<strong>除以 \(1-\beta_1^t\)，就是将历史梯度的累计权重归一化，消除初始零值使统计量偏小的影响。</strong> 二阶矩采用相同方式修正，只是统计对象变为梯度的平方。

上述例子使用恒定梯度。实际梯度变化时，修正得到的是历史梯度的加权平均，并非当前梯度。随着更新次数增加，两个分母逐渐接近 1，修正影响减弱。

##### 2.2.3 参数更新：按历史梯度尺度调整幅度 {#3参数更新按历史梯度尺度调整幅度}

设 \(\eta_t\) 为学习率，\(\epsilon\) 为避免分母过小的正数。不考虑权重衰减时：

\[
\theta_{t+1}=\theta_t-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
\]

用 \(\sqrt{\hat v_t}+\epsilon\) 对 \(\hat m_t\) 逐元素缩放，再乘以学习率，得到参数改变量。历史梯度尺度较大的元素，其更新会受到更强的缩放约束。最终更新幅度由分子、分母及学习率共同决定。

**优化器状态**：一阶矩、二阶矩与更新步数跨迭代保留，用于后续更新。它们不属于模型权重，也不直接参与前向计算。每轮重新创建优化器会丢失这些历史状态。

#### 2.3 AdamW（Adam with Decoupled Weight Decay，解耦权重衰减的 Adam） {#3-adamwadam-with-decoupled-weight-decay解耦权重衰减的-adam}

若将权重衰减项加入梯度，Adam 会将它与损失梯度一起进行历史统计和缩放，使衰减效果受到各参数历史梯度的影响。因此，这种处理不等同于直接按固定比例衰减权重。

<strong class="note-key-point">AdamW 将权重衰减与梯度更新分开处理，使衰减项不参与梯度的历史统计。</strong> 损失梯度按 Adam 的规则处理，权重则单独按比例衰减。

##### 2.3.1 参数更新 {#1参数更新}

设 \(\theta_t\) 为当前参数，\(\eta_t\) 为学习率，\(\lambda\) 为权重衰减系数。\(\hat m_t\)、\(\hat v_t\) 按 Adam 的规则计算，但只使用损失产生的梯度，不加入 \(\lambda\theta_t\)：

\[
\theta_{t+1}=(1-\eta_t\lambda)\theta_t
-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
\]

先将当前参数乘以 \(1-\eta_t\lambda\)，施加减小权重绝对值的作用，再减去 Adam 计算得到的梯度更新量。两部分共同决定最终参数值，因此更新后的权重绝对值不一定减小。

**与 Adam 的区别**：两者的差异在于权重衰减的处理方式，而非一阶矩、二阶矩的计算方式。关闭权重衰减且其他配置相同时，两者的更新规则一致。

**优化器状态**：与 Adam 相同，维护一阶矩、二阶矩和更新步数，无需为权重衰减额外保存一组历史统计。

#### 2.4 Muon（MomentUm Orthogonalized by Newton–Schulz，基于 Newton–Schulz 正交化的动量优化器） {#4-muonmomentum-orthogonalized-by-newtonschulz基于-newtonschulz-正交化的动量优化器}

神经网络中的许多权重以矩阵形式存在。Adam 按元素调整更新幅度，但不会直接以均衡奇异值为目标处理更新矩阵。奇异值反映了矩阵在不同方向上的作用强度。当更新集中在少数较强方向时，其他方向的相对贡献可能较弱。

<strong class="note-key-point">Muon 对动量形成的更新矩阵做近似正交化，减弱不同矩阵方向之间的尺度差异。</strong> 它利用矩阵结构调整更新，而不是分别统计每个元素的梯度平方。

##### 2.4.1 构造更新方向 {#1构造更新方向}

设 \(W_t\) 为当前权重矩阵，\(G_t\) 为损失对该矩阵的梯度，\(\mu\) 为动量系数，\(M_{t-1}\) 为历史动量，初始状态取零。以标准动量的简化形式表示：

\[
M_t=\mu M_{t-1}+G_t,
\qquad
Q_t\approx\operatorname{Orth}(M_t).
\]

将历史动量与当前梯度组合，计算得到动量矩阵 \(M_t\)，再对其进行近似正交化，得到更新方向 \(Q_t\)。常见实现会在动量步骤使用额外的前瞻修正，具体形式可能不同。

##### 2.4.2 正交化的含义 {#2正交化的含义}

一个矩阵可以分解为若干方向及各方向上的强度，这些强度由奇异值表示。若 \(M_t=U\Sigma V^\top\)，理想的正交化方向为 \(UV^\top\)：保留方向结构，将奇异值的尺度拉齐。

实际计算通常用少量 Newton–Schulz 迭代近似这一变换，避免每次执行完整的奇异值分解。有限次迭代并不保证得到精确的正交矩阵。<strong>这里处理的是更新方向，不是强制模型权重本身正交。</strong>

##### 2.4.3 参数更新与适用范围 {#3参数更新与适用范围}

设 \(\eta_t\) 为学习率，\(s\) 为实现中采用的尺度系数。不考虑权重衰减时，更新可概括为：

\[
W_{t+1}=W_t-\eta_t sQ_t.
\]

将近似正交化后的方向按学习率和尺度系数缩放，得到权重矩阵的改变量。尺度系数的选择、动量变体和权重衰减方式需与具体训练实现对应。

Muon 主要用于隐藏层的二维权重矩阵。偏置、归一化参数，以及嵌入层和输出层等通常由 AdamW 等优化器处理，因此常采用分组配置。

**优化器状态**：主要维护动量矩阵，不保存 Adam 那样的逐元素二阶矩。近似正交化会增加矩阵运算，实际训练收益取决于模型结构与实现。

### 3. LLM 中的损失函数与优化目标 {#llm训练目标}

LLM 的训练不仅关注能否生成正确、连贯的文本，也关注回答是否有帮助、是否符合人类偏好。交叉熵通过目标文本提供学习信号，而回答之间的偏好比较和奖励评分则提供了另一类反馈。利用这些反馈训练模型，需要将其转化为相应的损失，并根据需要加入分布约束。

- **回答偏好**：利用同一问题下回答的优劣比较，使模型更倾向于生成受偏好的回答，DPO 是其中一种方法。
- **奖励评分**：利用生成结果的评分，引导模型提高高奖励回答的生成概率，PPO、GRPO 等训练算法采用了这一思路。
- **分布约束**：通过 KL 散度限制模型偏离参考分布的程度，或使学生模型学习教师模型的输出分布。

这些目标也涉及 LLM 后训练中的偏好优化与强化学习，通过回答偏好、奖励反馈及分布约束，进一步调整模型的生成行为。
