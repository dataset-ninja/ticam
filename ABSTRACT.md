**TICaM: A Time-of-Flight In-Car Cabin Monitoring Dataset** is a time-of-flight dataset of car in-cabin images providing means to test extensive car cabin monitoring systems based on deep learning methods. The authors provide depth, RGB, and infrared images of front car cabin that have been recorded using a driving simulator capturing various dynamic scenarios that usually occur while driving. For dataset they provide ground truth annotations for 2D and 3D object detection, as well as for instance segmentation.

Note, similar **TICaM: A Time-of-Flight In-Car Cabin Monitoring Dataset** dataset is also available on the [DatasetNinja.com](https://datasetninja.com/):

- [TICaM Synthetic: A Time-of-Flight In-Car Cabin Monitoring Dataset](https://datasetninja.com/ticam-synthetic)

## Dataset description

With the advent of autonomous and driver-less vehicles, it is imperative to monitor the entire in-car cabin scene in order to realize active and passive safety functions, as well as comfort functions and advanced human-vehicle interfaces. Such car cabin monitoring systems typically involve a camera fitted in the overhead module of a car and a suite of algorithms to monitor the environment within a vehicle. To aid these monitoring systems, several in-car datasets exist to train deep leaning methods for solving problems like driver distraction monitoring, occupant detection or activity recognition. The authors present TICaM, an in-car cabin dataset of 6.7K real time-of-flight depth images with ground truth annotations for 2D and 3D object detection, and semantic and instance segmentation. Their intention is to provide a comprehensive in-car cabin depth image dataset that addresses the deficiencies of currently available such datasets in terms of the ambit of labeled classes, recorded scenarios and provided annotations; all at the same time.

<img src="https://github.com/dataset-ninja/ticam/assets/120389559/5b05542f-9fa0-4ddb-a14c-f1810e0bb123" alt="image" width="800">

<span style="font-size: smaller; font-style: italic;">Real images. IR-image, depth image with 2D bounding box annotation, segmentation mask.</span>

A noticeable constraint in existing car cabin datasets is the omission of certain frequently encountered driving scenarios. Notably absent are situations involving passengers, commonplace objects, and the presence of children and infants in forward and rearward facing child seats. The authors have conscientiously addressed this limitation by capturing a diverse array of everyday driving scenarios. This meticulous approach ensures that our dataset proves valuable for pivotal automotive safety applications, such as optimizing airbag adjustments. Using the same airbag configuration for both children and adults can pose a fatal risk to the child. Therefore, it is imperative to identify the occupant class (*person*, *child*, *infant*, *object*, or *empty*) for each car seat and determine the child seat configuration (forward-facing FF or rearward-facing RF). The dataset also includes annotations for both driver and passenger ***activity***. Recognizing ***activity*** is not only vital for innovative contactless human-machine interfaces but can also be integrated with other modalities, such as driver gaze monitoring. This integration enables a robust estimation of the driver's state, activity, awareness, and distraction critical factors for hand-over maneuvers in conditional or highly automated driving scenarios. A notable omission in widely used in-car cabin datasets is their lack of multi-modality. 

In addressing this gap, the authors present a comprehensive set of depth, RGB, and infrared images, emphasizing the inclusion of 3D data annotations. These images were systematically captured within a driving simulator, employing a Kinect Azure device securely positioned near the rear-view mirror. This setup offers a more practical viewpoint compared to other datasets and a mounting position that can be faithfully replicated within actual cars. The utilization of Time-of-Flight depth modality is chosen for its distinct advantages. Depth images offer enhanced privacy as subjects remain unidentifiable, exhibit increased resilience to variations in illumination and color, and facilitate natural background removal. The authors furnish 2D and 3D bounding boxes, along with class and instance masks, for depth images. These annotations can be directly applied for training on infrared images and, with some pre-processing, on RGB images as well. This is possible since the relative rotation and translation between depth and RGB cameras are known and provided. Furthermore, the authors enhance the dataset's comprehensiveness by including annotations for activity recognition tasks, ensuring a thorough coverage of input modalities and ground truth annotations.

## Data Acquisition

For data recording, the authors used an in-car cabin test platform. It consists of a realistic in-car cabin mock-up, equipped with a wide angle projection system for a realistic driving experience. A Kinect Azure camera with a wide field of view is mounted at the rear-view mirror position for 2D and 3D data recording. The camera is set to record at 30fps with 2×2 binning. The captured data consists of RGB, depth and IR amplitude images. To ensure a wide range of variability in the dataset, we adjust the seat positioning of the driver and passenger seats in the driving simulator.

<img src="https://github.com/dataset-ninja/ticam/assets/120389559/672f7287-26c5-40bb-9948-f2f0d17a3d0c" alt="image" width="800">

<span style="font-size: smaller; font-style: italic;">The data capturing setup equipped with a wideangle projection system, car front seats and a Kinect Azure camera in the front.</span>

The authors initiate their dataset creation process by outlining specific use cases or scenarios they aim to include. Subsequently, they document these scenarios with the participation of 13 individuals, comprising 4 females and 9 males. The scenarios encompass various configurations: 
1) Sole presence of the driver, 
2) Presence of both driver and passenger, 
3) Presence of the driver and an object, 
4) Presence of the driver with an empty Forward Facing Child Seat (FF), 
5) Presence of the driver with an empty Rearward Facing Infant Seat (RF), 
6) Presence of the driver with an occupied Forward Facing Child Seat (FF), 
7) Presence of the driver with an occupied Rearward Facing Infant Seat (RF), 
8) Sole presence of an object.

To ensure a well-orchestrated dataset, the authors choreograph specific actions for both the driver and passenger. These instructions are shared with participants prior to recording driving sequences. For instance, drivers engage in actions such as sitting, normal driving, looking left while turning the wheel, turning right, and more. On the other hand, passengers perform actions such as conversing with the driver, retrieving items from the dashboard, and the like. In total, the authors delineate 20 distinct actions for both drivers and passengers.

For each participant or pair of participants (in cases where both driver and passenger are involved), multiple sequences are recorded with varying positions of car seats. This deliberate variation ensures that any occupant classification and activity recognition model trained on the dataset remains robust across different seat configurations.

<img src="https://github.com/dataset-ninja/ticam/assets/120389559/0f82c33e-da3d-4312-ae30-047b66865d83" alt="image" width="800">

<span style="font-size: smaller; font-style: italic;">Depth and IR images of different driving scenarios provided in TICaM.</span>


In addition to diversifying the scenarios, the authors introduce variations in the appearance of individuals by incorporating different clothing accessories such as jackets and hats. To address practical considerations, the authors opt for human dolls as substitutes for children and infants in scenarios where a single person is driving with a Forward Facing (FF) or Rearward Facing (RF) seat securely positioned on the passenger seat. In conjunction with the dolls, three Forward Facing (FF) and three Rearward Facing (RF) seats are employed, each set in different orientations such as sunshade up/down or handle up/down.

<img src="https://github.com/dataset-ninja/ticam/assets/120389559/5ec8b42a-6209-411d-8226-c977adfbdf10" alt="image" width="800">

<span style="font-size: smaller; font-style: italic;">Example human dolls and child seats used for recording scenarios with children and infants on the passenger seat.</span>

##  Data Format

* **Depth Z-image.** The depth image is undistorted with a pixel resolution of 512 × 512 pixels and captures a 105◦ × 105◦ FOV. The depth values are normalized to [1mm] resolution and clipped to a range of [0, 2550mm]. Invalid depth values are coded as ‘0’. Images are stored in 16bit PNG-format.

* **IR Amplitude Image.** Undistorted IR images from the depth sensor are provided in the same format as the depth image above.

* **RGB Image.** Undistorted color images are saved in PNG-format in 24bit depth. While the synthetic RGB images have the same resolution and field of view as the corresponding depth images (512 × 512), the real recorded RGB images have a higher resolution of 1280×720 pixels, but a lower field of view of 90◦×59◦ FOV.

* **2D bounding boxes.** For each depth image, the 2D boxes are defined by the top-left and bottom-right corners of the box, its class label and a flag ***low remission*** which is set to 1 for objects which are either blac kor very reflective or both, and therefore are barely visible in the depth image.

* **Pixel segmentation masks.** For each depth image two corresponding masks are generated: instance mask and class mask. The pixel intensities in these
masks correspond to the class ID in the class mask and the instance ID for a certain class in the instance mask.

* **Activity annotations.** For all sequences with people in the driver or passenger seat, we provide a .csv file describing the activities performed throughout the sequence. Each .csv contains the ***activity*** ID, ***activity*** name, ***person*** ID, a label either driver or passenger to specify if the action is performed by the driver or he passenger, the starting frame number of the action, the ending frame and the ***duration*** of that action in frames.