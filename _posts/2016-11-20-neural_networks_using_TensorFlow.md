---
layout: post
title: G-DRAGON
date: 2016-11-20 
tags:   
---

## 介绍
G-DRAGON（权志龙、권지용），1988年8月18日出生于韩国首尔，韩国男歌手，男子演唱组合BIGBANG队长，所属经纪公司YG Entertainment词曲制作人。
2001年，因参与特别企划专辑《大韩民国Hip-Hop Flex》而出道。2006年，作为组合BIGBANG成员身份正式出道   。2009年，发行第一张专辑《Heartbreaker》  。2012年9月15日，发行第二张专辑《One of a kind》 。2013年9月，推出专辑《COUP D'ETAT》；8月31日至9月1日，在首尔奥林匹克体操竞技场举办演唱会  ；同年，获得MAMA“年度最佳歌手”、“最佳男歌手”、“最佳舞蹈表演男歌手”、“最佳音乐录影带”奖项。2014年，创作发行与TAEYANG合唱的单曲《Good Boy》  。2015年9月1日，随组合BIGBANG正式发行专辑《MADE SERIES》   。2016年7月，随组合BIGBANG一起拍摄的电影《BIGBANG MADE》正式上映 。2017年6月，发布个人同名专辑《Kwon Ji Yong》   ；同月10日，在首尔举办solo演唱会，开启19个城市个人巡演


中文名权志龙                                                                职    业 歌手、Rapper 、制作人
外文名 G-Dragon/지드래곤/권지용/Kwon Ji Yong                                 毕业院校  庆熙大学
别    名 GD/鸡涌                                                            经纪公司   YG Entertainment
国    籍 韩国                                                               代表作品:谎言、一天一天、Heartbreaker、One Of A Kind、Crooked
民    族 朝鲜族                                                             主要成就 :2007年MKMF最佳编曲赏 2008年韩国十大作曲家                                                                  
星    座 狮子座                                                                      2008年最佳魅力先生 2009年MAMA年度最佳专辑等
血    型 A型                                                                         
身    高 178cm                                                              音乐类型K-Pop、Hip-Hop
体    重 57kg 
出生地 韩国首尔 
出生日期 1988年8月18日

<div align="center">
	<img class="lb_mainimg" unselectable="on" alt="" data-link="http://www.win4000.com/wallpaper_detail_129642.html" data-imgid="5912300835d7598eb9ada2236466acfa" data-src="http://p3.so.qhimgs1.com/t0187ac16044f19f433.jpg" src="http://pic1.win4000.com/wallpaper/c/593a4259d8ea6.jpg" style="top: 0px; opacity: 1; width: 595.2px; height: 372px; left: 0px;">

</div> 


### 目录

* [大家是怎样评价他的？](#When-to-apply-neural-net)
* [G dragon 权志龙，到底是怎样一种存在？](#solve-problems)
* [了解图像数据和主流的库来解决问题](#popular-libraries)
* [什么是 TensorFlow？](#What-is-TensorFlow)
* [TensorFlow 一个 典型 的 “ 流 ”](#A-typical-flow)
* [在 TensorFlow 中实现 MLP](#MLP)
* [gd依然吊打小鲜肉？数据了解一下](#Limitations-of-TensorFlow)
* [TensorFlow 与其他库](#vs-libraries)
* [从这里去哪里？](#Where-to-go-from-here)


### <a name="When-to-apply-neural-net"></a>大家是怎样评价他的？


“So… I am Kwon Ji Yong and also G-DragonActually… Many people know me as D-Dragon. That’s true. Talking like this, it feels a bit weird to introduce myself, I am not sure how to properly introduce.The person you see on the screen right now is, to someone a son, a friend, Love, or maybe a favorite star, celebrity… but the important thing is I am not even sure who I really am. I am always trying to look good when I dress up as “GD” like this. Well… but the reality is sometimes it feels too heavy on me.Still, I am kind of afraid to take it off. I may seem like… It might seem like I am living the “good life”, but these days… I don’t know. At least in front of you guys, no… at least in this very moment… I wanted to be more honest. I cannot show you everything but showing you who Kwon Ji Yong REALLY is for the first time. I wish I can be someone who still shines without all these shiny things on. I’ve been living as G-Dragon until now, but now, I want to live being “KWON JI YONG”.I don’t know what you want me to be but what you see right now is everything. So… I am not sure how I look to you, You are not sure what I’m saying, right? Uh… (going crazy) Who am I? Do you know who you are? ”然后他唱了新专《权志龙》里面那首Superstar.VCR之后有一个地方他现场用英文独白，可是场子里太吵了，时不时就有人高喊“G-Dragon,we love you.” 不重要的，他早就明白。“权志龙是谁？”这个答案并不重要。来看演唱会的人，只想看到那个舞台上闪闪发光的G-Dragon，所以他只需要扮演那个G-Dragon就够了。“权志龙是谁？”他自己也不知道了。整张新专他发挥了自己vocal的能力，节奏舒缓，浅吟低唱。他在回忆曾经的自己，也在审视现在的自己。一个内向而孤独的人，一个内心翻腾着岩浆的人，一个戴着面具生活的人，只有两条路可以走——疯或死。
* **一些人会利用 `神经网络` 解决复杂的问题，如图像处理，**  `神经网络` 属于一类代表学习的算法，这些算法可以把复杂的问题分解为简单的形式，使他们成为可以理解的（或 “可表示”），就像吞咽食物之前的咀嚼，让我们更容易吸收和消化。这个分解的过程如果使用传统的算法来实现也可以，但是实现过程将会很困难。

* **选择适当类型的 `神经网络` ，来解决问题，**  每个问题的复杂情况都不一样，所以数据决定你解决问题的方式。 例如，如果问题是序列生成的问题，`递归神经网络` 更合适。如果它是图像相关的问题，想更好地解决可以采取 `卷积神经网络`。

* **最后最重要的就是 `硬件` 要求了，硬件是运行 `神经网络` 模型的关键。** 神经网被 “发现” 很久以前，他们在近年来得到推崇的主要的原因就是计算资源更好，能更大发挥它的光芒，如果你想使用 `神经网络` 解决这些现实生活中的问题，那么你得准备购买一些高端的硬件了😆！

### <a name="solve-problems"></a>大家都写的很多，我就随便说几点


* BIGBANG出道快八年了，而且在第二年在韩国已经家喻户晓了
* 他们有五个人，三个唱歌的，大成,太阳，胜利，两个说rap的，GD（也就是权志龙 队长）和TOP
* TOP和GD的rap让我觉得别的组合的歌的rap纯粹是为了rap而rap的
* TOP和GD的rap让我觉得别的组合的歌的rap纯粹是为了rap而rap的
* TOP和GD的rap让我觉得别的组合的歌的rap纯粹是为了rap而rap的
* 他们每个人确实可以出一张专辑，而且专辑都参与了制作
* 他们确实拿到了几乎能在韩国拿到的所有音乐奖项（分猪肉什么的就不说，不管存在与否，奖已经颁了
* 他们确实出过一些负面新闻，但是运气不错，都解决的还好，没有留下什么后遗症
* 他们在日本已经小有成就，也已经引起了美国音乐界的注意，ps:GD，鸟叔，比伯貌似三个人要合作一首新歌，虽然比伯最近名声不太好
* 他们的歌曲风格种类比较多，但是应该和华语圈的音乐风格有点格格不入，再加上造型独特，所以在我天朝网络文化中还是处于不利的局面
* 未完待续————————
　　
`
### <a name="popular-libraries"></a>了解图像数据和主流的库来解决问题
在他所在的行业里是顶级精英。在那个圈子里是别人无法复制的奇迹。作为普通人他也很真诚努力，渴望成功，遭遇过人生绝境但最终顽强挺身。作为被广泛崇拜也被广泛非议的特立独行的人，用作品和教养打翻每一张对他充满恶意的嘴脸。对我来说他是人生榜样，除了爸爸妈妈之外的精神支柱，在世界崩塌之时的灵魂依靠。即使他并不知晓我的存在，但他已经完成了他的梦想，那就是成为别人的梦。虽不能说是神一样的存在，但是一直鼓励vip前进的人啊，是永远值得我们学习的榜样啊，在被世人误会抄袭时，会勇敢面对！面对丑闻，会默默伤心，更珍惜家人，伙伴！面对不断的称赞，依然会90度鞠躬！在夜店玩嗨，不会特别避开“摄像头”让大家了解他的更多面！也会默默做着公益活动！在舞台，自信的样子！
目前有种把他当家人的感觉
不想说了，贴几张图吧
<img src="https://pic1.zhimg.com/80/e870f54b7db5d4b6dc45a04a195328d8_hd.png" data-rawheight="1057" data-rawwidth="750" class="origin_image zh-lightbox-thumb lazy" width="750" data-original="https://pic1.zhimg.com/e870f54b7db5d4b6dc45a04a195328d8_r.jpg" data-actualsrc="https://pic1.zhimg.com/e870f54b7db5d4b6dc45a04a195328d8_b.png">
舞台上霸气的他，平常有礼貌温暖的他，以及最让粉丝心水的像个小孩子小傻子一样的他。
<img src="https://pic1.zhimg.com/80/v2-46a03776f912ce5d877179be2ffb8ed4_hd.jpg" data-rawwidth="3240" data-rawheight="1080" class="origin_image zh-lightbox-thumb lazy" width="3240" data-original="https://pic1.zhimg.com/v2-46a03776f912ce5d877179be2ffb8ed4_r.jpg" data-actualsrc="https://pic1.zhimg.com/v2-46a03776f912ce5d877179be2ffb8ed4_b.jpg">
<img src="https://pic1.zhimg.com/80/v2-204cad8962e7762dd29f9b9d076d9acc_hd.jpg" data-rawwidth="3240" data-rawheight="1080" class="origin_image zh-lightbox-thumb lazy" width="3240" data-original="https://pic1.zhimg.com/v2-204cad8962e7762dd29f9b9d076d9acc_r.jpg" data-actualsrc="https://pic1.zhimg.com/v2-204cad8962e7762dd29f9b9d076d9acc_b.jpg">
<img src="https://pic1.zhimg.com/80/v2-730c79a5d2ed775109bcb22e78142630_hd.jpg" data-rawwidth="3240" data-rawheight="1080" class="origin_image zh-lightbox-thumb lazy" width="3240" data-original="https://pic1.zhimg.com/v2-730c79a5d2ed775109bcb22e78142630_r.jpg" data-actualsrc="https://pic1.zhimg.com/v2-730c79a5d2ed775109bcb22e78142630_b.jpg">
祝福你，我们鸡涌。
____
_____
________
________________

GDragon和权志龙感觉是两个人，哈哈，他们给人的感觉是不一样的。  GDragon是舞台上那个霸气十足的人，是很认真很努力完美主义的歌手，是有些忧郁敏感，思想让人捉摸不透的人，是让人敬佩，心疼的巨星。 权志龙是温暖爱笑羞涩可爱的人，会默默做一些暖心的事情，对别人对粉丝都特别好的人，是讨人喜欢的人，是心中重要的存在。

### <a name="A-typical-flow"></a>gd依然吊打小鲜肉？数据了解一下
　<img src="https://pic3.zhimg.com/v2-5b76699fd9f84f35c257b93295409896_r.jpg" class="ImageView-img" alt="preview" style="width: 654px; transform: translate3d(613px, 304.992px, 0px) scale3d(1.10092, 1.10092, 1); opacity: 1;">
 <img src="https://pic4.zhimg.com/80/v2-94d73799de03da5796d3f55de596c3b3_hd.jpg" data-rawwidth="720" data-rawheight="288" class="origin_image zh-lightbox-thumb lazy" width="720" data-original="https://pic4.zhimg.com/v2-94d73799de03da5796d3f55de596c3b3_r.jpg" data-actualsrc="https://pic4.zhimg.com/v2-94d73799de03da5796d3f55de596c3b3_b.jpg">
 <img src="https://pic4.zhimg.com/80/1d3fa299e54560958646feaf422fad43_hd.png" data-rawwidth="984" data-rawheight="1749" class="origin_image zh-lightbox-thumb lazy" width="984" data-original="https://pic4.zhimg.com/1d3fa299e54560958646feaf422fad43_r.jpg" data-actualsrc="https://pic4.zhimg.com/1d3fa299e54560958646feaf422fad43_b.png">

* **还有谁**，服不服？？？



### <a name="Limitations-of-TensorFlow"></a>TensorFlow 的限制

* 尽管 TensorFlow 是强大的，它仍然是一个低水平库，例如，它可以被认为是机器级语言，但对于大多数功能，您需要自己去模块化和高级接口，如 keras
* 它仍然在继续开发和维护，这是多么👍啊！
* 它取决于你的硬件规格，配置越高越好
* 不是所有变成语言能使用它的 API 。
* TensorFlow 中仍然有很多库需要手动导入，比如 OpenCL 支持。

上面提到的大多数是在 TensorFlow 开发人员的愿景，他们已经制定了一个路线图，计划库未来应该如何开发。

### <a name="vs-libraries"></a>TensorFlow 与其他库

　　TensorFlow 建立在类似的原理，如使用数学计算图表的 Theano 和 Torch，但是随着分布式计算的额外支持，TensorFlow 更好地解决复杂的问题。 此外，TensorFlow 模型的部署已经被支持，这使得它更容易用于工业目的，打开一些商业的三方库，如 Deeplearning4j ，H2O 和 Turi。 TensorFlow 有用于 Python，C ++ 和 Matlab 的 API 。 最近还出现了对 Ruby 和 R 等其他语言的支持。因此，TensorFlow 试图获得通用语言支持。

### <a name="Where-to-go-from-here"></a>从这里去哪里？

　　以上你看到了如何用 TensorFlow 构建一个简单的神经网络，这段代码是为了让人们了解如何开始实现 TensorFlow。 要解决更复杂的现实生活中的问题，你必须在这篇文章的基础上在调整一些代码才行。

　　许多上述功能可以被抽象为给出无缝的端到端工作流，如果你使用 scikit-learn ，你可能知道一个高级库如何抽象“底层”实现，给终端用户一个更容易的界面。尽管 TensorFlow 已经提取了大多数实现，但是也有更高级的库，如 TF-slim 和 TFlearn。

### 参考资源
* [TensorFlow 官方库](https://github.com/tensorflow/tensorflow) 
* Rajat Monga（TensorFlow技术负责人） [“TensorFlow为大家”](https://youtu.be/wmw8Bbb_eIE)  的视频
* [一个专用资源的策划列表](https://github.com/jtoy/awesome-tensorflow/#github-projects)  

### 关于原文

感谢原文作者 [Faizan Shaikh](https://www.analyticsvidhya.com/blog/author/jalfaizy/) 的分享，
这篇文章是在 [An Introduction to Implementing Neural Networks using TensorFlow](https://www.analyticsvidhya.com/blog/2016/10/an-introduction-to-implementing-neural-networks-using-tensorflow/) 的基础上做的翻译和局部调整，如果发现翻译中有不对或者歧义的的地方欢迎在下面评论里提问，我会加以修正 。

<img class="lb_mainimg" unselectable="on" alt="" data-link="http://www.duitang.com/blog/?id=601879332" data-imgid="dc096f9b820f0c948f0f0caba1ba406f" data-src="http://p3.so.qhimgs1.com/t01c12423370ad2eb33.jpg" src="http://img3.duitang.com/uploads/item/201607/12/20160712080213_LkGYF.thumb.700_0.jpeg" style="top: 0px; opacity: 1; width: 263.03px; height: 372px; left: 0px;">

<br>
转载请注明：[潘柏信的博客](http://baixin) » [使用 TensorFlow 实现神经网络](http://baixin.io/2016/11/neural_networks_using_TensorFlow/)  

