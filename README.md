#my code is inspired by https://github.com/ibab/tensorflow-wavenet
and https://github.com/soobinseo/waveneta

# Using pyTorch to implement the WaveNet for vocal separation
# remove the background music from a song

  - vstrain.ipynb
     - all the main code is in this file
  - plotLoss.ipynb
    - when you training the model, you can use this file to visualize model's loss trend 
    - babysit the model
  - playTorch.ipynb
    - just begin to learn pytorch, try some functions
  - vsFromTwoFile.ipynb
    - ongoing task, use other method to remove the music from songs
  - vsCorpus
    - training set, testing set and some results 
 - lossRecord
   - model will write the loss file to this folder

![one of good loss image](./lossRecord/loss.png)

