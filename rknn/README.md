```                
 _______     ___  ____   ____  _____  ____  _____  
|_   __ \   |_  ||_  _| |_   \|_   _||_   \|_   _| 
  | |__) |    | |_/ /     |   \ | |    |   \ | |   
  |  __ /     |  __'.     | |\ \| |    | |\ \| |   
 _| |  \ \_  _| |  \ \_  _| |_\   |_  _| |_\   |_  
|____| |___||____||____||_____|\____||_____|\____|     
```
## Usage
1. Please download pre-trained model following README.md of root path
2. Freeze model to pb file using 
```
rknn/save_pb.py
```

3. Transform to rknn format 
```
rknn/pb_to_rknn.py -i ./data/tusimple_test_image/0.jpg -r ./rknn/lanenet.rknn
```

4. Evaluate performance for the model in NPU 
```
rknn/rknn_perf.py -i ./data/tusimple_test_image/0.jpg -r ./rknn/lanenet.rknn
```

5. Run test in 1808
```
rknn/rknn_test.py -i ./data/tusimple_test_image/0.jpg -r ./rknn/lanenet.rknn
```
