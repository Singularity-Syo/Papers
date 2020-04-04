## 3 Deep Learning Libraries and Resources
A remarkable aspect of advances in deep learning so far is the enormous number of resources developed and shared by the community. 
These range from tutorials, to overviews of research papers, to open sourced code.
Throughout this survey, we will reference some of these materials in the topic specific sections, but we first list here a few general very useful frameworks and resources.

深度学习的一个显著进步是社区开发和共享了大量的资源，从教程到研究论文综述，再到开源代码，应有尽有。在整个调查过程中，将在主题的特定部分中引用其中的一些材料，但首先在这里列出一些非常有用的框架和资源。

### **Software Libraries for Deep Learning**
Arguably the two most popular code libraries for deep learning are PyTorch (with a high level API called Lightning) and TensorFlow (which also offers Keras as a high level API.) 
Developing and training deep neural network models critically relies on fast, parallelized matrix and tensor operations (sped up through the use of Graphical Processing Units) and performing automatic differentiation for computing gradients and optimization (known as autodiff.) 
Both PyTorch and TensorFlow offer these core utilities, as well as many other functions. 
Other frameworks include Chainer, ONNX, MXNET and JAX. 
Choosing the best framework has been the source of significant debate. 
For ramping up quickly, programming experiences closest to native Python, and being able to use many existing code repositories,
PyTorch (or TensorFlow with the Keras API) may be two of the best choices.

### **用于深度学习的软件库**

可以说用于深度学习的两个最流行的代码库是 PyTorch (带有高级 API Ligntning) 和 TensforFlow (提供 Keras 作为一种高级 API)。
开发和训练深层神经网络模型主要依赖于快速的、并行化的矩阵和张量运算(通过使用图形处理单元 GPU 加快速度) ，以及用于计算梯度和优化 (称为 autodiff) 的自动微分
Pytorch 和 TensorFlow 都提供这些核心实用程序以及许多其他功能。
其他框架包括 Chainer、 ONNX、 MXNET 和 JAX。
选择最好的框架一直是争论的焦点。
为了快速增长，最接近原生 Python 的编程体验，并且能够使用许多现有的代码库,
PyTorch (或者使用 Keras API 的 TensorFlow) 可能是两个最佳选择。

### **Tutorials**
1. https://course.fast.ai/ 
   fast.ai provides a free, coding-first course on the most important deep learning techniques as well as an intuitive and easy to use code library, https://github.com/fastai/fastai, for model design and development. 
2. https://towardsdatascience.com/ contains some fantastic tutorials on almost every deep learning topic imaginable, crowd sourced from many contributors. 
3. Many graduate deep learning courses have excellent videos and lecture notes available online, such as http://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/ for Deep Learning and Neural Networks, or the more topic specific Stanford‘s CS224N NLP with Deep Learning. 
   A nice collection of some of these topic specific lectures is provided at https://github.com/Machine-Learning-Tokyo/AI_Curriculum. 
   There are also some basic interactive deep learning courses online, such as https://github.com/leriomaggio/deep-learning-keras-tensorflow.

### **教程**
1. https://course.fast.ai/ fast.ai 提供免费的编程优先课程，介绍最重要的深度学习技术，以及直观易用的代码库，用于模型设计和开发 https://github.com/fastai/fastai。
2. https://towardsdatascience.com/ 包含了几乎所有你能想到的深度学习主题的教程，这些教程都是从许多贡献者那里获得的。
3. 许多研究生深度学习课程都有优秀的视频和课堂讲稿可以在网上找到，比如深度学习和神经网络 http://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/ ，或者斯坦福大学更具体的 CS224N NLP with Deep Learning。https://github.com/machine-learning-tokyo/ai_curriculum 提供了这些专题讲座的精选集。网上还有一些基本的交互式深度学习课程，比如 https://github.com/leriomaggio/deep-learning-keras-tensorflow。

### **Research Overviews, Code, Discussion**
1. https://paperswithcode.com/ 
   This excellent site keeps track of new research papers and their corresponding opensourced code, trending directions and displays state of the art results (https://paperswithcode.com/sota) across many standard benchmarks.
2. Discussion of deep learning research is very active on Twitter. http://www.arxiv-sanity.com/top keeps track of some of the top most discussed papers and comments. 
3. https://www.reddit.com/r/MachineLearning/ is also a good forum for research and general project discussion. 
4. https://www.paperdigest.org/conference-paper-digest/ contains snippets of all the papers in many different top machine learning conferences. 
5. IPAM (Institute for Pure and Applied Mathematics) has a few programs e.g. https://www.ipam.ucla.edu/programs/workshops/new-deep-learning-techniques/?tab=schedule and https://www.ipam.ucla.edu/programs/workshops/deep-learning-and-medical-applications/?tab=schedule with videos overviewing deep learning applications in science.

### **研究概况，代码，讨论**
1. https://paperswithcode.com/ 这个优秀的网站跟踪新的研究论文和他们相应的开源代码，趋势方向和显示在许多标准基准上的最先进的结果 ( https://paperswithcode.com/sota)
2. 关于深度学习研究的讨论在 Twitter 上非常活跃。 http://www.arxiv-sanity.com/top 保存了一些最热门的讨论论文和评论。
3. https://www.reddit.com/r/machinelearning/ 也是一个研究和一般项目讨论的好论坛。
4. https://www.paperdigest.org/conference-paper-digest/ 包含了许多顶级机器学习会议上所有论文的片段。
5. IPAM (Institute for Pure and Applied Mathematics)有一些项目，比如 https://www.IPAM.ucla.edu/programs/workshops/new-deep-learning-techniques/?tab=schedule 和 https://www.IPAM.ucla.edu/programs/workshops/deep-learning-and-medical-applications/?tab=schedule ，它们的视频覆盖了科学领域的深度学习应用。
   
### **Models, Training Code and Pretrained Models** 
As we discuss later in the survey, publicly available models, training code and pretrained models are very useful for techniques such as transfer learning. 
There are many good sources of these, here are a few that are especially comprehensive and/or accessible:
1. Pytorch and TensorFlow have a collection of pretrained models, found at https://github.com/tensorflow/models and https://pytorch.org/docs/stable/torchvision/models.html.
2. https://github.com/huggingface Hugging Face (yes, that really is the name), offers a huge collection of both pretrained neural networks and the code used to train them. 
   Particularly impressive is their library of Transformer models, a one-stop-shop for sequential or language applications.
3. https://github.com/rasbt/deeplearning-models offers many standard neural network architectures, including multilayer perceptrons, convolutional neural networks, GANs and Recurrent Neural Networks.
4. https://github.com/hysts/pytorch_image_classification does a deep dive into image classification architectures, with training code, highly popular data augmentation techniques such as cutout, and careful speed and accuracy benchmarking. 
   See their page for some object detection architectures also.
5. https://github.com/openai/baselines provides implementations of many popular RL algorithms.
6. https://modelzoo.co/ is a little like paperswithcode, but for models, linking to implementations of neural network architectures for many different standard problems.
7. https://github.com/rusty1s/pytorch_geometric. 
   Implementations and paper links for many graph neural network architectures.

### **模型，训练代码和预训练模型**

正如我们在后面的调查中所讨论的，公开可用的模型、培训代码和预训练模型对于诸如迁移学习这样的技术非常有用。
这里有很多很好的资源，这里有一些是特别全面和 / 或容易获得的:
1. Pytorch 和 TensorFlow 收集了一系列预训练模型，可以在 https://github.com/TensorFlow/models 和 https://Pytorch.org/docs/stable/torchvision/models.html 找到。
2. https://github.com/huggingface 拥抱面孔(是的，这确实是它的名字) ，提供了一个预训练的神经网络和用来训练它们的代码的庞大集合。特别令人印象深刻的是他们的 Transformer 模型库，一个一站式的顺序或语言应用。
3. https://github.com/rasbt/deeplearning-models 提供了许多标准的神经网络体系结构，包括多层感知器、卷积神经网络、生成对抗网络和循环神经网络。
4. https://github.com/hysts/pytorch_image_classification 对图像分类架构进行了深入研究，使用了训练代码、非常流行的如剪切的数据增强技术，以及谨慎的速度和准确性基准测试。也可以通过他们的页面来了解一些目标检测架构。
5. https://github.com/openai/baselines 提供了许多流行的 RL 算法的实现。
6. https://modelzoo.co/ 有点像 paperwithcode，但实际上是模型方面，连接到神经网络架构的实现来解决许多不同的标准问题。
7. https://github.com/rusty1s/pytorch_geometric. 许多图形神经网络架构的实现和论文。

#### **Data Collection, Curation and Labelling Resources**
A crucial step in applying deep learning to a problem is collecting, curating and labelling data. 
This is a very important, time-intensive and often highly intricate task (e.g. labelling object boundaries in an image for segmentation.) 
Luckily, there are some resources and libraries to help with this, for example https://github.com/tzutalin/labelImg, https://github.com/wkentaro/labelme, https://rectlabel.com/ for images and https://github.com/doccano/doccano for text/sequential data.

#### **数据收集、管理和标签资源**

将深度学习应用于一个问题的关键步骤是收集、整理和标记数据。
这是一项非常重要、耗时且常常非常复杂的任务 (例如在图像中标记物体边界以便进行分割)
幸运的是，有一些资源和库可以帮助解决这个问题，例如图像领域： https://github.com/tzutalin/labelImg、 https://github.com/wkentaro/labelme、https://rectlabel.com/ 和文本 / 序列数据： https://github.com/doccano/doccano。

#### **Visualization, Analysis and Compute Resources** 
When training deep neural network models, it is critical to visualize important metrics such as loss and accuracy while the model is training. 
Tensorboard https://www.tensorflow.org/tensorboard (which works with Pytorch and TensorFlow) is a very popular framework for doing this.
Related is the colab effort https://colab.research.google.com/notebooks/welcome.ipynb, which, aside from providing a user-friendly, interactive way for model development and analysis (very similar to jupyter notebooks) also provides some (free!) compute resources.

#### **可视化、分析和计算资源**
在训练深层神经网络模型时，模型训练过程中的损失和准确度等重要指标的可视化至关重要。
https://www.TensorFlow.org/Tensorboard 是一个非常流行的用于这一方面的框架。

相关的还有 colab 的努力 https://colab.research.google.com/notebooks/welcome.ipynb ，它除了提供一个用户友好的，交互式的方式为模型开发和分析 (非常类似于 Jupyter Notebooks) 还提供了一些(免费!) 计算资源。
