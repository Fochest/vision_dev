# vision_dev

## Scripts

- ### split_data.sh
   This script splits a directory full of images into multiple numbered directories with 100 images each

   - #### Usage
      `./split_data.sh <source directory>`
      - source directory: [string] The directory storing the original set of image
      
## Sloth
Sloth is an image tagging tool.

### Configuration
Provided, is the sloth configuration file, `robosub_config.py`, to specialize sloth for Robosub use. Included, are the required annotation conversions for any frameworks used by robosub.

- ### Annotation types
   - #### Darknet
      Darknet has its own specific format which requires multiple files, one per image. For this, we have created a special *.darknet format to bridge the gap between sloth's one file system to darknet's multiple file system. The *.darknet format is as follows
      
      ```
      ---
      <class name> : <class number>
      class_name_1 : 0
      class_name_2 : 1
            :
            :
      ---
      >> "filename_1.jpg"
      <class number> <x_position_ratio> <y_position_ratio> <width_ratio> <height_ratio> //Darknet format line
      <<
      >> "filename_2.png"
      <<
          :
          :
       ```
       - class name: [string] The name of the class within sloth for the tag
       - class number: [integer] The number assotiated with the class name used within darknet
       - x_position_ratio: [0.0...1.0] The ratio between the x position of the center of the rectangle and the image width
       - y_position_ratio: [0.0...1.0] The ratio between the y position of the center of the rectangle and the image height
       - width_ratio: [0.0...1.0] The ratio between the width of the rectangle and the image width
       - height_ratio: [0.0...1.0] The ratio between the height of the rectangle and the image height
       
        Utilize the `sloth_to_darknet.py` script to generate the darknet files from the above annotation format.

### Scripts
- ### sloth_to_darknet.py
   Converts from the *.darknet format specified above to multiple files within an output directory and a training list used with Darknet
   - #### Usage
      `./sloth_to_darknet.py -f <filename> -o <output directory> [-t <training list filename>]`
      - filename: [string] The filepath to the *.darknet file from which to extract information.
      - output directory: [string] The directory into which the darknet compatible annotation files are output.
      - (optional) training list filename: [string] The filename of the output training list
- ### sloth_to_tfrecord.py
   Converts json format files into tenserflow format.
   - #### Usage
      `./sloth_to_tfrecord.py <config_file> <output directory> --crossvalidation <crossvalidation> --test <testratio> --evaluation <evaluationratio>`
      - input [STRING]: resourcename from REST resource or json-file ending in .json
         - .json for offline-usage
      - outputdir [STRING]: folder in which the tfrecord files will be written + labelMap
      - crossvalidation [INT] (default 1): determines amounts of training- & test-datasets that will be generated using the crossvalidation algorithm
         - eg: 10 -> evaluationratio is substracted from all files - the remaining are used to create X sets. Has higher priority than the testratio parameter. crossvalidation == 1 -> data is split into [trainingset / testset / validationset] according to testratio & evaluationratio 
      - testratio [FLOAT] (default 0.1): determines the ratio of files will be used as a test record
         - eg: 0.1 -> 10% of the labelled files will be written to test.tfrecord
      - evaluationratio [FLOAT] (default 0.2): determines the ratio of files will be used as an evaluation record
         - eg: 0.1 -> 10% of the labelled files will be written to eval.tfrecord
