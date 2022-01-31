
# Omni-eye

This project was created to make 3D pose estimation possible using stereovision and AI models.  Anyone that is in a field that requires pose estimation of a subject can benefit from this project.
 

## Authors

- [Thomas](https://www.github.com/tomlogan-dev1)
- [Pushaan](https://www.github.com/shaan1999s)
- [Alvin](https://www.github.com/mhyo4125)



## Installation

The first requirement for this project is to own a [Nvidia Jetson Xavier NX](https://www.amazon.com/NVIDIA-Jetson-Xavier-Developer-812674024318/dp/B086874Q5R/ref=sr_1_2?crid=2XIX0VD5MYD2J&keywords=nvidia+jetson+xavier+nx&qid=1638136020&qsid=131-2276662-2444818&sprefix=nvidia+jetson+xavier%2Caps%2C201&sr=8-2&sres=B086874Q5R%2CB083ZL3X5B%2CB09BPXZMB3%2CB094F9BR61%2CB084DSDDLT%2CB08MQBY79Y%2CB08HR6ZBYJ%2CB07T5BNF14%2CB08J157LHH%2CB08F743RGG%2CB091YXMVHT%2CB06XPFH939%2CB07ZYJYGZ5%2CB07NSRK2DL%2CB08PDZ68D7%2CB07L8YGDL5&srpt=PERSONAL_COMPUTER) 
.  The code we wrote for this project is also configured to be used with the [IMX219-160](https://www.amazon.com/Jetson-Nano-Camera-IMX219-160-8-Megapixels/dp/B07T43K7LC/ref=sr_1_1_sspa?keywords=nvidia+jetson+camera&qid=1638136101&sr=8-1-spons&psc=1&smid=A2SA28G0M1VPHD&spLa=ZW5jcnlwdGVkUXVhbGlmaWVyPUEyMFA4UzQ5NUNRNERQJmVuY3J5cHRlZElkPUEwNDA2NTY2MkZXSkUwSTc5WTBKMyZlbmNyeXB0ZWRBZElkPUEwNzkyMzc1MTZYMUFTU1Y2TFVLRyZ3aWRnZXROYW1lPXNwX2F0ZiZhY3Rpb249Y2xpY2tSZWRpcmVjdCZkb05vdExvZ0NsaWNrPXRydWU=) camera module.


#### The first step to installation is to configure the Jetson

1. Install and upload the JetPack.  Follow this [guide](https://www.stereolabs.com/blog/getting-started-with-jetson-xavier-nx/)
2. Install Pytorch and Torchvision. Follow this [guide](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048)
3. Install trt_pose

```bash
git clone https://github.com/NVIDIA-AI-IOT/trt_pose
cd trt_pose
sudo python3 setup.py install
```
4. Download the [DenseNet](https://drive.google.com/file/d/13FkJkx7evQ1WwP54UmdiDXWyFMY1OxDU/view) and [ResNet](https://drive.google.com/file/d/1XYDdCUdiF2xxx4rznmLb62SdOUZuoNbd/view) pretrained models.

#### The next step is to download [Blender](https://www.blender.org/download/) on your host PC
In this project we are using version 2.93.6




## Getting Started

Once all of the dependencies are downloaded onto the Nvidia Jetson, we can start by running the [video_recoder.py]().

When this code is running it will ask for an input.  To start recording a video, that you want to turn into an animation, input the letter 'r', and then press esc to end the recording.  If you'd like to view the video you just recorded simply press 'p'.

When you are satisfied with your recorded video, simply run [pose_recorder.py]().  This will automatically created a file that containes the data the Blender needs to 3D render your custom animation sequence.

Now that your animation sequence file is created open [create_animation.blend]() on your host computer.
Run the code and see the 3D skeleton of your animation in the top-right of the screen.  Once the animation sequence has been created in Blender you can link it to any avatar you like and created 3D renderings.


