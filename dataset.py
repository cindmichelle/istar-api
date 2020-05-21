import os
from mrcnn import utils
import skimage
from skimage.transform import rotate
import json
import numpy as np

class IstarDataset(utils.Dataset):
    def load_istar(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # TODO : Add more classes if test success (Goal, Actor, Quality).
        self.add_class("shapes", 1, "Actor")
        self.add_class("shapes", 2, "Goal")
        self.add_class("shapes", 3, "Quality")
        self.add_class("shapes", 4, "Task")
        self.add_class("shapes", 5, "Resource")

        # self.add_class("type", 6, "Role")
        # self.add_class("type", 7, "Agent")
        # self.add_class("type", 8, "Unknown")



        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotationsDict = json.load(open(os.path.join(dataset_dir, "annotations.json")))
        annotationsArr = list(annotationsDict.values())  # don't need the dict keys


        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotationsArr if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            # print('filename ', a['filename'])
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            objects = [s['region_attributes'] for s in a['regions'].values()]

            ## NOTE : this contains label of the objects
            class_ids_name = [int(n['element_name']) for n in objects]
            # class_ids_type = [int(n['element_type']) for n in objects]

			# load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            # print("multi_numbers=", multi_numbers)
            # num_ids = [n for n in multi_numbers['number'].values()]
            # for n in multi_numbers:

            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path, plugin='pil')
            isLandscape= a['file_attributes']['landscape']
            height, width = image.shape[:2]

            if isLandscape :
              if a['filename'] == '20200218_165243_noise_test.jpg' :
                print(a['filename'], height, width)

              if width < height :
                if a['filename'] == '20200218_165243_noise_test.jpg' :
                  print('should be rotate')
                image = rotate(image, 90, resize=True)
                height, width = image.shape[:2]

                if a['filename'] == '20200218_165243_noise_test.jpg' :
                  print('after rotate (h,w) : ', height, width)

                # print("should be rotate into landscape")

            else : #if portrait
              if height < width :
                image = rotate(image, 90, resize=True)
                height, width = image.shape[:2]
                # print("should be rotate into portrait")

            self.add_image(
                "shapes",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
				class_ids=class_ids_name)

        #     self.add_image(
        #         "type",
        #         image_id=a['filename'],  # use file name as a unique image id
        #         path=image_path,
        #         width=width, height=height,
        #         polygons=polygons,
				# class_ids=class_ids_type)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]

        # print("IMAGE INFO :", image_info)
        if image_info["source"] != "shapes" :
            return super(self.__class__, self).load_mask(image_id)
        class_ids = image_info['class_ids']
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            if p['name'] == 'polyline':
              rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            elif p['name'] == 'circle':
              rr, cc = skimage.draw.circle(p['cy'], p['cx'], p['r'])
            # rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        #class_ids=np.array([self.class_names.index(shapes[0])])
        # print("info['class_ids']=", info['class_ids'])
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
