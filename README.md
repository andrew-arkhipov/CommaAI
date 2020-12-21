# CommaAI Speed Challenge - INCOMPLETE
Sample code for commaAI speed challenge.

<br>

<strong>Task:</strong> predicting speed of a car from dashcam footage.</br>
<strong>Approach:</strong> using dense optical flow and feature matching to track feature across consecutive frames in order to create a mapping system. Avoids dynamic features (e.g. other  cars) by implementing a pre-trained CNN. Using the research papers in `Research Papers` to help guide approach.</br>

<strong>Previous approach:</strong> used `visualizer.py` to produce average changes in distance of image features over consecutive frames, which were then fed into an recurrent CNN. Model wasn't able to pick up on any patterns due to lack of numerous, salient static features in consecutive frames. Code is in `scrapped`.

<br>

### Virtual Environment

It is highly recommended to use a Python virtual environment when running this script. Run the following commands in the root directory of the project.
```
python3 -m venv <env-folder-name>
```

To activate that virtual environment, use the below command:
```
source <env-folder-name>/bin/activate
```

After activating, to install the dependencies run:
```
pip3 install -r requirements.txt
```

### Running the code
For sparse flow, run:
```
python odometry.py
```

For dense flow (incomplete), run:
```
python dense.py
```
