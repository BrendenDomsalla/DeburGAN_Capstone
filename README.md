# DeburGAN_Capstone
My personal attempt to replicate the results of DeblurGAN using tensorflow as the library rather than Pytorch

Run main.py and open the gradio app in a browser, and from there just
upload the image into the browser

Run train.py to train the model on your own dataset. If you just want my 
dataset its the REDS Dataset from Nvidia. The link is 
https://seungjunnah.github.io/Datasets/reds
I didn't provide my dataset for obvious reasons, I didn't feel like 
watching git try to handle 80gb of image files, these is a single file 
of testing images for when you download this


Retraining using this architecture:
If you want to train this on your own basic image set, use DataLoader 
not DataLoader2; 2 is designed to handle this specific dataset and the 
issues I had with it and its specific file structure

If you want to