# SKillNav

Official implementation of the paper:

**Breaking Down and Building Up: Mixture of Skill-Based Vision-and-Language Navigation Agents**

[Paper and Appendix](https://arxiv.org/abs/2508.07642)



## **1. Matterport3D Simulator Setup**

We use the **latest** version of the [Matterport3D Simulator](https://github.com/peteanderson80/Matterport3DSimulator) (not v0.1).

We use python==3.9 and 

- Install system packages (EGL/OSMesa/OpenGL)

```bash
sudo apt-get update
# Core dependencies
sudo apt-get install -y libjsoncpp-dev libepoxy-dev libglm-dev libopencv-dev \
                        libegl1 libegl1-mesa-dev libgl1-mesa-dev libtiff-dev
# For OSMesa / GLEW builds
sudo apt-get install -y libosmesa6 libosmesa6-dev libglew-dev
```

- Conda packages

```bash
conda install -c conda-forge cmake gdal libtiff -y
# If you hit missing C++ symbols at runtime, this newer libstdc++ helps:
conda install -c conda-forge libstdcxx-ng -y
```

- Build the simulator 

```bash
# Replace [your_python_bin_path] with the absolute path to Python in the 'vlnde' env
# Example: /home/$USER/miniconda3/envs/vlnde/bin/python
cd Matterport3DSimulator
mkdir -p build && cd build
cmake -DEGL_RENDERING=ON -DPYTHON_EXECUTABLE=[your_python_bin_path] ..
make -j
```

- Add to your environment:

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```



## **2. Construct Data**

### **R2R skill-specific data**

Download from: [Google Drive Link](https://drive.google.com/drive/folders/1ZYCi2yJAHqk96Ptud6sqQRtWJalIyig7?usp=sharing)

Place the extracted files into:

```
ScaleVLN/datasets/R2R/annotations/
```



## **3. Baselines**

- [**ScaleVLN**](https://github.com/wz0919/ScaleVLN)
- [**VLN-SRDF**](https://github.com/wz0919/VLN-SRDF)

Both baselines are included in this repo.



## **4. **Training & Testing

- Train

```bash
cd ./ScaleVLN/map_nav_src
bash ./scripts/train_r2r_b16_mix_vertical.sh
```

- Test

```bash
cd ./ScaleVLN/map_nav_src
bash ./scripts/test_r2r_b16_moe-top1.sh
```



## **5. Citation**

```
@misc{ma2025breakingbuildingupmixture,
      title={Breaking Down and Building Up: Mixture of Skill-Based Vision-and-Language Navigation Agents}, 
      author={Tianyi Ma and Yue Zhang and Zehao Wang and Parisa Kordjamshidi},
      year={2025},
      eprint={2508.07642},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2508.07642}, 
}
```

