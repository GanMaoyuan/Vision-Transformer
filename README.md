# [《An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale》](https://arxiv.org/abs/2010.11929)论文阅读笔记

## 图像序列化

用 ![](https://latex.codecogs.com/svg.latex?x\in\mathbb{R}^{H\times{}W\times{}C}) 来表征一个图像，这个图像被分成 ![](https://latex.codecogs.com/svg.latex?H\times{}W) 个像素，![](https://latex.codecogs.com/svg.latex?H) 行 ![](https://latex.codecogs.com/svg.latex?W) 列，高（Height）即为 ![](https://latex.codecogs.com/svg.latex?H) ，宽（Width）即为 ![](https://latex.codecogs.com/svg.latex?W) 。<br>
每个像素由 ![](https://latex.codecogs.com/svg.latex?C) （如3）个通道（Channel）表征，每个通道表征一种原色（如RGB，即Red、Green、Blue中的某一个）。<br>
通道值为 ![](https://latex.codecogs.com/svg.latex?0) 时，该通道亮度为 ![](https://latex.codecogs.com/svg.latex?0) （不发光）；通道值最大（在8-bit图中对应 ![](https://latex.codecogs.com/svg.latex?2^8-1) ）时，该通道亮度最大。<br>
例如R通道（Red），通道值最大时该通道表征为最亮的红色。<br>
像素的颜色为混合色，由 ![](https://latex.codecogs.com/svg.latex?C) 个通道进行综合表征。在特殊情况下，![](https://latex.codecogs.com/svg.latex?C) 个通道的值均为 ![](https://latex.codecogs.com/svg.latex?0) ，此时像素表征为纯黑色，![](https://latex.codecogs.com/svg.latex?C) 个通道的值均达到最大时像素表征为纯白色。<br>
图像的分辨率与分割出来的像素的数量有关，用行数、列数来表示，即 ![](https://latex.codecogs.com/svg.latex?(H,W)) 。在这张分割像素图上的左上角取一个小块（Patch），这个小块由 ![](https://latex.codecogs.com/svg.latex?P\times{}P) 个像素组成（ ![](https://latex.codecogs.com/svg.latex?P) 行 ![](https://latex.codecogs.com/svg.latex?P) 列），分辨率即为 ![](https://latex.codecogs.com/svg.latex?(P,P)) 。![](https://latex.codecogs.com/svg.latex?P) 的值一般取为 ![](https://latex.codecogs.com/svg.latex?H) 与 ![](https://latex.codecogs.com/svg.latex?W) 的公因数，这样的话就可以从左上角开始，从左到右、从上到下分割出 ![](https://latex.codecogs.com/svg.latex?N=HW/P^2) 个完整的小块。<br>
每个小块包含 ![](https://latex.codecogs.com/svg.latex?P^2) 个像素，每个像素又由 ![](https://latex.codecogs.com/svg.latex?C) 个通道值（实数）来表征，那么就可以用一个含有 ![](https://latex.codecogs.com/svg.latex?P^2C) 个元素的实数张量 ![](https://latex.codecogs.com/svg.latex?x_P\in\mathbb{R}^{1\times{}P^2C}) 来直接表征这个小块。<br>
输入到模型的是整张图而非单个小块，因此，实际输入是一个含有 ![](https://latex.codecogs.com/svg.latex?N\times{}P^2C) 个元素的实数张量 ![](https://latex.codecogs.com/svg.latex?x_p\in\mathbb{R}^{N\times{}P^2C}) ，即一个 ![](https://latex.codecogs.com/svg.latex?N) 行 ![](https://latex.codecogs.com/svg.latex?P^2C) 列的矩阵，直接表征 ![](https://latex.codecogs.com/svg.latex?N) 个小块，也就是整张图。<br>
这个张量可以理解为表征了一个含有 ![](https://latex.codecogs.com/svg.latex?N) 个token的序列，每个token的表征维度大小暂时为 ![](https://latex.codecogs.com/svg.latex?P^2C) 。这与一个普通的文本输入序列没有区别。
<br><br><br><br><br>

## 嵌入层

嵌入维度（Dimension）大小为 ![](https://latex.codecogs.com/svg.latex?D) ，注意与在自注意力层之后的前馈层内部才短暂出现的隐藏（Hidden）维度的大小 ![](https://latex.codecogs.com/svg.latex?H) 作区分，前者小于后者。<br>
在序列刚刚进入模型时的词（图像小块）嵌入阶段，需要让 ![](https://latex.codecogs.com/svg.latex?x_p^i\in\mathbb{R}^{1\times{}P^2C}) 乘一个可训练的投影矩阵 ![](https://latex.codecogs.com/svg.latex?E\in\mathbb{R}^{P^2C\times{}D}) ，投影为 ![](https://latex.codecogs.com/svg.latex?x_p^iE\in\mathbb{R}^{1\times{}D}) 。整个图像小块序列经过该阶段得到的输出称为图像小块嵌入，具体为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?z_0=\begin{bmatrix}x_{\text{class}}\\x_p^1E\\x_p^2E\\{}\vdots\\x_p^NE\end{bmatrix}+E_{\text{pos}},\quad{}x_{\text{class}}\in\mathbb{R}^{1\times{}D},\quad{}E_{\text{pos}}\in\mathbb{R}^{(N+1)\times{}D},">
</p>

![](https://latex.codecogs.com/svg.latex?z_0) 的下标 ![](https://latex.codecogs.com/svg.latex?0) 表示第 ![](https://latex.codecogs.com/svg.latex?0) 层，即“嵌入层”，区别于第 ![](https://latex.codecogs.com/svg.latex?1) 层、第 ![](https://latex.codecogs.com/svg.latex?2) 层、... ![](https://latex.codecogs.com/svg.latex?\space) 、第 ![](https://latex.codecogs.com/svg.latex?L) 层等编码器层。<br>
![](https://latex.codecogs.com/svg.latex?z_0\in\mathbb{R}^{(N+1)\times{}D}) 是一个 ![](https://latex.codecogs.com/svg.latex?N+1) 行 ![](https://latex.codecogs.com/svg.latex?D) 列的矩阵，由 ![](https://latex.codecogs.com/svg.latex?N+1) 个形状为 `[1,D]` 的张量沿行方向拼接而成，其第0个子张量 ![](https://latex.codecogs.com/svg.latex?z_0^0=x_{\text{class}}) 经过 ![](https://latex.codecogs.com/svg.latex?L) 个编码器层的变换，在第 ![](https://latex.codecogs.com/svg.latex?L) 个编码器层的输出（编码器的最终输出）当中的状态为 ![](https://latex.codecogs.com/svg.latex?z_L^0=y_{\text{class}}) ，用于表征图像内容的类别，并且将直接作为一个分类头（由一个前馈层或一个单一的线性投影层实现）的输入，模型根据分类头的输出判断图像内容类别的名称（如“树”、“车”、“房子”等）。<br>
![](https://latex.codecogs.com/svg.latex?z_0^0=x_{\text{class}}) 与 ![](https://latex.codecogs.com/svg.latex?z_L^0=y_{\text{class}}) 的上标 ![](https://latex.codecogs.com/svg.latex?0) 表示序列的位置 ![](https://latex.codecogs.com/svg.latex?0) 处，序列的实际内容元素占据位置 ![](https://latex.codecogs.com/svg.latex?1) 、位置 ![](https://latex.codecogs.com/svg.latex?2) 、![](https://latex.codecogs.com/svg.latex?\cdots) ![](https://latex.codecogs.com/svg.latex?\space) 、位置 ![](https://latex.codecogs.com/svg.latex?L)。<br>
通过引入这样一个可训练的类别张量，就可以让模型在训练过程中将图像内容的类别信息注入这个类别张量，从而在最后只需将这个经过层层变换的类别张量作为分类头的唯一输入，即可得到输出的类别结果。<br>
可训练的 ![](https://latex.codecogs.com/svg.latex?E_{\text{pos}}) 用于在训练过程中注入各个图像小块的二维位置信息。
<br><br><br><br><br>

## 多头自注意力

“自注意力”当中“自”这个字的逻辑在于查询 ![](https://latex.codecogs.com/svg.latex?q) 、键 ![](https://latex.codecogs.com/svg.latex?k) 、值 ![](https://latex.codecogs.com/svg.latex?v) 均由长度为 ![](https://latex.codecogs.com/svg.latex?L) 的输入序列 ![](https://latex.codecogs.com/svg.latex?z\in\mathbb{R}^{L\times{}D}) 自身经过三个不同的投影来产生，具体为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?q=zW_q\in\mathbb{R}^{L\times{}D_h},\quad{}k=zW_k\in\mathbb{R}^{L\times{}D_h},\quad{}v=zW_v\in\mathbb{R}^{L\times{}D_h},">
</p>

其中 ![](https://latex.codecogs.com/svg.latex?h) 表示自注意力的头（head）数，![](https://latex.codecogs.com/svg.latex?D_h=D/h) 。<br>
序列中的元素（token或图像小块）![](https://latex.codecogs.com/svg.latex?i) 对序列全体元素发起查询，其中元素 ![](https://latex.codecogs.com/svg.latex?j) 对它而言，注意力（Attention）权重为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?A_{ij}=\text{softmax}\left(\frac{q_ik_j^\top}{\sqrt{D_h}}\right),">
</p>

其中 ![](https://latex.codecogs.com/svg.latex?q_i) 取自 ![](https://latex.codecogs.com/svg.latex?q) 的第 ![](https://latex.codecogs.com/svg.latex?i) 行，![](https://latex.codecogs.com/svg.latex?k_j^\top) 取自 ![](https://latex.codecogs.com/svg.latex?k^\top) 的第 ![](https://latex.codecogs.com/svg.latex?j) 列。定义矩阵 ![](https://latex.codecogs.com/svg.latex?B) 的第 ![](https://latex.codecogs.com/svg.latex?i) 行第 ![](https://latex.codecogs.com/svg.latex?j) 列元素

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?B_{ij}=q_ik_j^\top,">
</p>

这个形式实际上符合矩阵乘运算的定义（ ![](https://latex.codecogs.com/svg.latex?X) 与 ![](https://latex.codecogs.com/svg.latex?Y) 的矩阵乘结果 ![](https://latex.codecogs.com/svg.latex?Z=XY) 的第 ![](https://latex.codecogs.com/svg.latex?i) 行第 ![](https://latex.codecogs.com/svg.latex?j) 列元素值等于 ![](https://latex.codecogs.com/svg.latex?X) 的第 ![](https://latex.codecogs.com/svg.latex?i) 行与 ![](https://latex.codecogs.com/svg.latex?Y) 的第 ![](https://latex.codecogs.com/svg.latex?j) 列的内积值），于是，可以直接判定

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?B=qk^\top.">
</p>

此外又由于

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?A_{ij}=\text{softmax}\left(\frac{B_{ij}}{\sqrt{D_h}}\right),">
</p>

于是可定义注意力权重矩阵

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{align*}A&=\text{softmax}\left(\frac{B}{\sqrt{D_h}}\right)\\{}&=\text{softmax}\left(\frac{qk^\top}{\sqrt{D_h}}\right)\\{}&\in\mathbb{R}^{L\times{}L}.\end{align*}">
</p>

对于 ![](https://latex.codecogs.com/svg.latex?A_{ij}) ，先确定行下标 ![](https://latex.codecogs.com/svg.latex?i) ，而列下标 ![](https://latex.codecogs.com/svg.latex?j) 从 ![](https://latex.codecogs.com/svg.latex?1) 迭代至 ![](https://latex.codecogs.com/svg.latex?L) ，

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?j\in[1,2,\cdots,L],">
</p>

因此注意力权重矩阵的第 ![](https://latex.codecogs.com/svg.latex?i) 行的 ![](https://latex.codecogs.com/svg.latex?L) 个矩阵元素值实际上就是序列的全体 ![](https://latex.codecogs.com/svg.latex?L) 个元素对序列元素 ![](https://latex.codecogs.com/svg.latex?i) 的相应权重，于是单个自注意力（Self-Attention，SA）头的输出即为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\text{SA}(z)=Av\in\mathbb{R}^{L\times{}D_h},">
</p>

其第 ![](https://latex.codecogs.com/svg.latex?i) 行的 ![](https://latex.codecogs.com/svg.latex?D_h) 个矩阵元素值就共同组成了序列的位置 ![](https://latex.codecogs.com/svg.latex?i) 处的上下文表征，并且在单个自注意力头当中，这个上下文表征一般专注于序列的某一方面的含义。对于文本序列而言可以是“语法结构”，对于图像（小块）序列而言可以是“画面风格”。不同的自注意力头的上下文表征专注于不同方面的含义。<br>
![](https://latex.codecogs.com/svg.latex?h) 个自注意力头（多头自注意力，Multihead Self-Attention，MSA）的运算是并行的，互不干扰，各自的自注意力输出再沿上下文表征维度拼接（沿列方向拼接），序列元素 ![](https://latex.codecogs.com/svg.latex?i) 的上下文表征维度的大小就从原先的 ![](https://latex.codecogs.com/svg.latex?D_h) 成倍增长到 ![](https://latex.codecogs.com/svg.latex?h\cdot{}D_h=D) ，上下文表征也就相应地变得更加丰富，对序列含义的理解也就更加全面。具体为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\text{MSA}(z)=\left[\text{SA}_1(z);\text{SA}_2(z);\cdots;\text{SA}_h(z)\right]W_{msa}\in\mathbb{R}^{L\times{}D},\quad{}W_{msa}\in\mathbb{R}^{(h\cdot{}D_h)\times{}D}.">
</p>

<br><br><br>

## 编码器层

一个编码器层包含一个多头自注意力层与一个紧随其后的前馈层（多层感知机，Multi-Layer Perceptron，MLP）。设

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?l\in[1,2,\cdots,L],">
</p>

以第 ![](https://latex.codecogs.com/svg.latex?l) 层为例，具体为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{align*}z_l'&=\text{MSA}\left(\text{Norm}(z_{l-1})\right)+z_{l-1},\\{}z_l&=\text{MLP}\left(\text{Norm}(z_l')\right)+z_l'.\end{align*}">
</p>

前馈层包含一个线性层、一个非线性层、一个线性层。<br>
前一个线性层将大小为 ![](https://latex.codecogs.com/svg.latex?D) 的嵌入维度投影到大小为 ![](https://latex.codecogs.com/svg.latex?H) 的隐藏维度（ ![](https://latex.codecogs.com/svg.latex?D<H) ），非线性层对全体隐藏维度元素的值进行非线性变换（如负值置0，正值不变），后一个线性层将隐藏维度投影回嵌入维度。
