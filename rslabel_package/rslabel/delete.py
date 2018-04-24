#!/usr/bin/python
# This script is designed to remove bad images given tar name and json file to be present

import argparse
import json
import os
import sys
import tempfile
import glob
import tarfile

# Main function
def app(args):
    base_name = os.path.splitext(os.path.basename(args.tarname))[0]
    print("Working on {}".format(base_name))
    os.rename(base_name + ".json", base_name + "-delete.json")
    os.remove(base_name + ".json")

    # Extract tar
    print("Extracting")
    with tarfile.TarFile(args.tarname) as tf:
        tf.extractall()
    os.remove(args.tarname)

    with open(base_name + '-delete.json', 'r') as f:
        json_contents = json.load(f)

    # Looping over to see if there are bad images, and see if the tar is complete
    for annotation in json_contents:
        try:
            ann = annotation['status']
            if ann == 'Bad':
                print("Removing {}".format(annotation['filename']))
                os.remove("{}/{}".format(base_name, annotation['filename']))
        except:
            pass
    annotations = []
    print("Creating new json")
    for f in sorted(glob.glob('{}/*.jpg'.format(base_name))):
        annotations.append({'annotations': [],
                            'class': 'image',
                            'filename': os.path.basename(f),
                            'unlabeled': True})
    os.remove(base_name + '-delete.json')
    with open("{}/{}".format(base_name, base_name + '.json'), 'w') as f:
        json.dump(annotations, f, indent=4)
    print("Putting everything back into tar")
    with tarfile.open(base_name + '.tar', "w") as tar:
        tar.add(base_name)
    print("Putting tar up for labeling")
    os.rename(base_name + '.tar', "../../new/{}".format(base_name + '.tar'))
    print("Done!")
if __name__ == '__main__':
    # Arguement handler
    parser = argparse.ArgumentParser(description='Utility for deleting bad images.')
    parser.add_argument('tarname', type=str, help='Name of the tar with bad images')
    parser.set_defaults(func=app)
    args = parser.parse_args()
    args.func(args)
