# my code is first inspired by
    https://github.com/ibab/tensorflow-wavenet
    https://github.com/soobinseo/wavenet
    https://github.com/f90/Wave-U-Net

# Paper
    WaveNet,deepmind paper https://arxiv.org/pdf/1609.03499.pdf
    facebook paper https://arxiv.org/pdf/1805.07848.pdf
    Spotify paper wave-U-net https://arxiv.org/pdf/1806.03185.pdf

#  domain confusion loss
- which is used by facebook paper
- I just implemented the domain confusion loss in https://github.com/ShichengChen/Domain-Adversarial-Training-of-Neural-Networks
- but I think I cannot use the network confusion loss for my problem, since I need the domain information when I generate the music without voice. I should keep the music have same type.

# Using pyTorch to implement the WaveNet for vocal separation
# remove the voice from songs

  - vstrain.ipynb
     - all the main code is in this file, you can see more comments on this file
  - vstrainTowloss.py(now I do not use this, I will continue to test this in the future)
     - see as above file except that label are instrument and voice
  - trainunet.py
     - use waveunet to train the model, not exact unet, biggest change in the file is use Short-time Fourier transform to deal with data
  - DatasetWaveUnet.py
    - use librosa stft to deal with data
  - readDataset3.py
    - use h5py to speed up, read instument file and voice file
  - transformData.py 
    - provides mu_law encode and decode functions(actually, pytorch have these functions)
  - wavenet.py
    - structure of wavenet. learned from https://arxiv.org/pdf/1609.03499.pdf
    - have dilated cnn layers, you can learn dilated from    https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    - It's not easy to train, since there are sigmoid and tanh
  - wavenet2.py
    - structure of wavenet. learned from https://arxiv.org/pdf/1805.07848.pdf
    - replace sigmoid and tanh by relu, remove element-wise multiplication.
    - add relu at the beginning of every residual blocks
    - very easy to train and the results are good. 
    - if the music is not hard, such as many repeat rhythm
    - I can get good result for only instrument music by only 3 epochs(about 3000 iterations) for one song.
    - if the music is hard, they take electro acoustics as instrument
    - I need to train for 10 epochs for one song
  - wavenet3.py
    - structure of wavenet. inspired by https://github.com/f90/Wave-U-Net
    - same as wavenet2 except that I use two loss, one loss for instument and the other one for voice
    - two loss indeed have help, for a easy song, only 3 epochs(3000 iterations), the result can be good. voice file + instrument file = mix file. In other words, I add a new restriction for my network 
  - unet.py
     - structure of unet. learned from https://arxiv.org/pdf/1806.03185.pdf
     - I remove dilated cnn layers by normal cnn layers, 
  - clean_ccmixter_corpus.ipynb, clean_ccmixter_corpus2.ipynb
     - transform ccmixter from (audio time series, either stereo or mono) to mono
     - save as h5 format by h5py
  - plotLoss.ipynb
    - when you training the model, you can use this file to visualize model's loss trend 
    - babysit the model
    - use linear regression to fit the loss
  - useAandBsimultaneously.ipynb
    - ongoing task, use other method to remove the music from songs
  - originalResults
    - training set, testing set and some results 
 - lossRecord
   - model will write the loss file to this folder

# Dataset
 - origin_mix.wav(train)
 - origin_vocal.wav(label)
 - pred_mix.wav(test) (for one song, I get a good result)
 - ccmixter corpus (for 50 songs, I still try to improve my model and learning strategy)

# Training data and testing data
- first half of the songs as training data, the last half of the songs as testing data.

# Data Augmentation
- multiply source signals with a factor chosen uniformly from the interval [0.7, 1.0] and set the input mixture as the sum of source signals, which is learned from Spotify paper. 
- The result is so good by using this type of data augmentation. Even the loss become bigger, however, the voice became less. The model can generalize better.

# Result for one song
- the model is trained on training set except for last 15 seconds
- bestResultonTestingSet.wav (testing set)
- bestResultonTrainingSet.wav (training set except for last 15 seconds)
- bestResultonValidation.wav (last 15 seconds for training set)
- there are still some noise and a little music on audios on validation and testing set

# Result for ccmixter(50 songs)
- If I only train few songs, the results will be also good. 
- If I train on the whole dataset, the results will become worse. 

# Generalization for ccmixter(50 songs)
- ccmixter has 3 Children's songs, two songs as training data and the other as testing data, the result on testing data is also very good even though is slightly worse than training data.
- Three rap songs can also generalize well
- Two songs have different background music and same lyrics(two same voice), generalization is ok, ok, but worse than above two situations
- first 45 songs for training and last 5 songs for testing, the results are still not good.

# Loss for one song
 - best loss: around 1

# Loss for ccmixter
 - around 3

# hyper-parameters
 - sampleSize=16000#recommended by facebook paper
 - sample_rate=16000#the length of audio for one second
 - quantization_channels=256 #discretize the value to 256 numbers
 - ~~dilations=[2**i for i in range(9)]*7#recommended by facebook paper~~
 - dilations=[2**i for i in range(9)]*5 for quicker test
 - residualDim=128#recommended by facebook paper
 - skipDim=512
 - initFilterSize=25#recommened by https://github.com/f90/Wave-U-Net, help me remove a lot of noise
 - other filterSize=3
 - learning rate=1e-3 adam, decay factor of 0.98 every 10000 samples#recommended by facebook paper
 
# Notice
 - if i set residual channel to 256, the loss will stuck into 4.5,(actually, loss returns from 3.5)
 - if you design custom model, you should use self.convs = nn.ModuleList() instead of self.convs = dict(). If you use the latter way, the pytorch cannot update the weight in the dict() 
 - A(background music) + B(voice) = C(mix music)

# ToDo
 - ~~better learning rate decay strategy, speed up the training process.~~
 - ~~bigger dataset()~~
 - try to use better method for training ccmixter corpus(50 songs)
 - Short-time Fourier transform