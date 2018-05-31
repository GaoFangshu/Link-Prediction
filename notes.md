数据：
(1) paper ID
(2) 发表年份
(3) title
(4) 作者
(5) 期刊名
(6) abstract

要注意！我们需要预测的点已经在网络中了！

已用feature

- [ ] title分词overlap
- [ ] 发表时间年数间隔
- [ ] 作者overlap
- [x] N-gram
- [ ] abstract, lowercase, no stopwords, with stemmer 
- [x] abstract, lowercase, no stopwords, without stemmer 
- [ ] word2vec
- [ ] journal name (dummy)
- [ ] journal name network, strength is the citation number
- [x] N-gram:_dice_dist
- [x] N-gram:_jaccard_coef
- [ ] 期刊之间有关联，期刊配对之间有引用数
- [ ] 作者的单位
- [ ] 有的作者是大牛，就容易被引
- [ ] 这个作者以前引用过，未来更有可能还是引用
- [ ] 单向图，通过时间调整，但这样涉及到了哪些变量是单向的？
- [ ] 比如我现在要引文章，我要进入这个网络，我最有可能从哪里进入？可以类比搜索（因素一：哪个网页（文章）跟我的搜索（我的文章）最像？因素二：哪个文章影响力更大，更容易被我看到？）
- [ ] 是否有同年内引用？
- [ ] PageRank，越早的点越有可能PR值大，不过也不妨一试吧。但两个点的PR关系又说明了什么呢？？？SimRank更靠谱点，但原来的SimRank是同级关系，我现在的想法是：**想要知道a是否引b，即a在b之后发表，那可以考察引b的文章c们（任何文章）和a的相似度。**
- [ ] 相似度的测量？
    （1）用之前的n-gram相似度
   - [ ] 与c们的均值
    （2）Network相似度
   - [ ] 与c们的共同好友个数（无向（同时又上下游） or 有向（上游或下游）都有道理），Jaccard
   - [ ] 
- [ ] a是否引用b，如果a有很多短路径能到b的话，则可能性高，但越长可能也就难引用了（也不一定，如果b经典文献）。总之Katz Index可以一试。
- [ ] 训练集的0表示没有关系，这与未知不是一个概念，需要在graph中体现吗


reference
[Stanford CS224W Projects](http://snap.stanford.edu/class/cs224w-2010/proj2010/)