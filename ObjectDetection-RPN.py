# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 18:57:42 2019

@author: PuneetPC
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe


try:
    tf.enable_eager_execution()
except ValueError:
    # Already executed.
    pass

# Local imports.
from utils.faster import (
    clip_boxes, rcnn_proposals, run_base_network, run_resnet_tail,
    generate_anchors_reference, sort_anchors
)
from utils.image import open_all_images, open_image, to_image
from utils.vis import (
    add_rectangle, draw_bboxes, draw_bboxes_with_labels, image_grid,
    pager, vis_anchors
)

images = open_all_images('D:/Courses/Deep_learning/project/object-detection-utils/images/')

image = images['woman']
to_image(image)

#loading checkpoint to the resnet. These already trained checkpoints 

with tfe.restore_variables_on_create('D:/Courses/Deep_learning/project/object-detection-utils/checkpoint/fasterrcnn'):
    feature_map = run_base_network(image)
    

ANCHOR_BASE_SIZE = 256
ANCHOR_RATIOS = [0.5, 1, 2]
ANCHOR_SCALES = [0.125, 0.25, 0.5, 1, 2]


def generate_anchors(feature_map_shape):
 
    anchor_reference = generate_anchors_reference(
        ANCHOR_BASE_SIZE, ANCHOR_RATIOS, ANCHOR_SCALES
    )
    
    OUTPUT_STRIDE =  16
    with tf.variable_scope('generate_anchors'):
        grid_width = feature_map_shape[2]  # width of the feature 
        grid_height = feature_map_shape[1]  # height of the feature
        x1 = tf.range(grid_width) * OUTPUT_STRIDE
        y1 = tf.range(grid_height) * OUTPUT_STRIDE
        x1, y1 = tf.meshgrid(x1, y1)

        x2 = tf.reshape(x1, [-1])
        y2 = tf.reshape(y1, [-1])

        shifts = tf.stack([x1, y1, x2, y2],axis=0)

        shifts = tf.transpose(shifts)
        all_anchors = ( np.expand_dims(anchor_reference, axis=0) + tf.expand_dims(shifts, axis=1))      
    
    
    print(anchor_reference.shape)
    print("as numpy array")
    print(anchor_reference)
    return all_anchors


anchors = tf.reshape(generate_anchors(feature_map.shape), [-1, 4])

print('Anchors (real image size):')
print(anchors.shape)
print(anchors.numpy())


def get_dimensions_and_center(bboxes):

    ctx,cty,width,height = tf.split(bboxes, [1,1,1,1], 1)
     
    return ctx,cty,width,height

def encode(anchors, bboxes):
    xb, yb,wb, hb = get_dimensions_and_center(bboxes)
    xr, yr,wr, hr = get_dimensions_and_center(anchors)
    
    deltax = tf.math.divide(tf.math.subtract(xb,xr),wr)
    
    deltay = tf.math.divide(tf.math.subtract(yb,yr),hr)
    
    deltaw = tf.math.log(tf.math.divide(wb,wr))
    
    deltah = tf.math.log(tf.math.divide(hb,hr))
    
    deltas = tf.concat([deltax,deltay,deltaw,deltah],1)
    print("deltas:")
    print(deltas)
    return deltas



def decode(anchors, deltas):
    dx, dy,dw, dh = get_dimensions_and_center(deltas)
    xr, yr,wr, hr = get_dimensions_and_center(anchors)
    
    xb = tf.math.add(tf.math.multiply(dw,wr),xr)
    
    yb = tf.math.add(tf.math.multiply(dy,hr),yr)
    
    wb = tf.math.multiply(tf.math.exp(dw),wr)
    
    hb = tf.math.multiply(tf.math.exp(dh),hr)
    
    bboxes = tf.concat([xb,yb,wb,hb],1)
    print("boxes:")
    print(bboxes)
    return bboxes


def run_rpn(feature_map):
    #    print(feature_map)
    conv, cls, regress = _instantiate_cls_model(feature_map)
    print(feature_map.shape)

    rpn_feature = conv(feature_map)

        # Then we apply separate conv layers for classification and regression.
    rpn_cls_prob = cls(rpn_feature)
    rpn_bbox_pred = regress(rpn_feature)

    rpn_cls_prob = tf.reshape(rpn_cls_prob, [-1, 2])
    rpn_bbox_pred = tf.reshape(rpn_bbox_pred, [-1, 4])
        ####

    return rpn_bbox_pred, rpn_cls_prob

def _instantiate_cls_model(feature_map):

        #this method calls the tensorflow methods
        #model = tf.keras.Sequential()
        #print(input_shape)
    conv=tf.keras.layers.Conv2D(
                filters=512,
                kernel_size=(3,3),
                activation='relu',
                input_shape=feature_map.shape,
                padding='same',
                name='rpn/conv'
    )
    cls=tf.keras.layers.Conv2D(
                filters=30,
                kernel_size=(1,1),
                activation='softmax',
                padding='same',
                name='rpn/cls_conv'
    )
    regress=tf.keras.layers.Conv2D(
                filters=60,
                kernel_size=(1,1),
                activation='relu', 
                padding='same',
                name='rpn/bbox_conv'
    )

    return conv, cls, regress

with tfe.restore_variables_on_create('D:/Courses/Deep_learning/project/object-detection-utils/checkpoint/fasterrcnn'):
    rpn_bbox_pred, rpn_cls_prob = run_rpn(feature_map)


    expected_preds = (
        feature_map.shape[1]
        * feature_map.shape[2]
        * len(ANCHOR_RATIOS)
        * len(ANCHOR_SCALES)
    )
    print(expected_preds)
    print(rpn_bbox_pred.shape[0])
    print(rpn_cls_prob.shape[0])


proposals = decode(anchors, rpn_bbox_pred)
print(proposals.shape)

scores = tf.reshape(rpn_cls_prob[:, 1], [-1])
print(scores)


def keep_top_n(proposals, scores, topn):
    

    print(proposals.shape)
    print(scores.shape)
    print(topn)
    
    topk_values, topk_indeces=tf.math.top_k(scores,k=topn,sorted=True)
    print(topk_values.shape)
    print(topk_indeces.shape)
    sorted_top_proposals = tf.gather(proposals,topk_indeces,axis=0)
    sorted_top_scores = topk_values
    return sorted_top_proposals, sorted_top_scores

topn = 3000

top_raw_proposals, top_raw_scores = keep_top_n(proposals, scores, topn)

# Filter the first `topn` proposals, as ordered by score.

def filter_proposals(proposals, scores):
    
    props = proposals.numpy()
    areas = (props[:, 2] - props[:, 0]) * (props[:, 3] - props[:, 1])
    print(areas)
    zeros = tf.zeros(areas.shape,tf.float32)
    bools = tf.greater(areas,zeros)
    print(bools.shape)
    print(proposals.shape)
    proposals = tf.boolean_mask(proposals,bools)
    scores = tf.boolean_mask(scores,bools)
    print(proposals)
    print(scores)
    ####

    return proposals, scores


# Filter proposals with negative areas.
proposals, scores = filter_proposals(proposals, scores)

# non-maximum suppression.
PRE_NMS_TOP_N = 12000

# We will use the `keep_top_n` function that you have implemented before!
proposals, scores = keep_top_n(proposals, scores, PRE_NMS_TOP_N)

# Final maximum number of proposals, as returned by NMS.
POST_NMS_TOP_N = 2000

# IOU overlap threshold for the NMS procedure.
NMS_THRESHOLD = 0.7


# You might find the following function useful for re-ordening the coordinates
# as expected by Tensorflow.
def change_order(bboxes):
    
    print(bboxes.numpy())
    output = tf.unstack(bboxes)
    print(len(output))
    for i in range(len(output)):
        tf_min, tf_max = tf.split(output[i],2)
        tf_min =tf.reverse(tf_min,axis=[0])
        tf_max =tf.reverse(tf_max,axis=[0])
        output[i]=tf.concat([tf_min,tf_max],0)
    
    bboxes = tf.stack(output)
    print(bboxes.numpy())
    ####
    
    return bboxes

#change_order(proposals)

def apply_nms(proposals, scores):
    
    print(proposals)
    proposals = change_order(proposals)
    nms_indexes = tf.image.non_max_suppression(proposals,scores,POST_NMS_TOP_N,NMS_THRESHOLD,score_threshold=float('-inf'))
    
    proposals = tf.gather(proposals,nms_indexes,axis=0)
    proposals = change_order(proposals)
    ####

    return proposals, scores

pre_merge_proposals, pre_merge_scores = proposals, scores
proposals, scores = apply_nms(proposals, scores)


def normalize_bboxes(proposals, im_shape):
    
    print(proposals)
    output = tf.unstack(proposals)
    print(len(output))
    for i in range(len(output)):
        tf_min, tf_max = tf.split(output[i],2)
        tf_min =tf.reverse(tf_min/tf.reduce_max(tf.abs(tf_min)),axis=[0])
        tf_max =tf.reverse(tf_max/tf.reduce_max(tf.abs(tf_max)),axis=[0])
        output[i]=tf.concat([tf_min,tf_max],0)
    
    bboxes = tf.stack(output)
    print(bboxes)
    print(bboxes.shape)
    ####

    return bboxes
#normalize_bboxes(proposals,(image.shape[1], image.shape[2]))


def roi_pooling(feature_map, proposals, im_shape, pool_size=7):
    
    zeros = tf.zeros(proposals.shape[0],tf.int32)
    croped_feature_map = tf.image.crop_and_resize(feature_map,proposals,zeros,(2*pool_size,2*pool_size),method='bilinear')
    
    pooled = tf.nn.max_pool(croped_feature_map,(1,2,2,1),(1,2,2,1),padding='VALID')
    print(pooled.shape)
    ####

    return pooled


pooled = roi_pooling(feature_map, proposals, (image.shape[1], image.shape[2]))


# We're finally ready to perform the classification, so load the class names.
with open('D:/Courses/Deep_learning/project/object-detection-utils/checkpoint/classes.json') as f:
    classes = json.load(f)
    
print(classes)

def run_rcnn(pooled, num_classes):
    resnet_output = run_resnet_tail(pooled)
    print(resnet_output.shape)
    
    global_avg_pooling = tf.reduce_mean(resnet_output,(1,2))
    print(global_avg_pooling.shape)
    
    rcnn_cls_prob = tf.layers.dense(global_avg_pooling,units=num_classes+1,name='rcnn/fc_classifier',activation='softmax')
    
    rcnn_bbox = tf.layers.dense(global_avg_pooling,units=num_classes*4,name='rcnn/fc_bbox')
    
    print(rcnn_cls_prob.shape)
    print(rcnn_bbox.shape)

    return rcnn_bbox, rcnn_cls_prob


with tfe.restore_variables_on_create('D:/Courses/Deep_learning/project/object-detection-utils/checkpoint/fasterrcnn'):
    bbox_pred, cls_prob = run_rcnn(pooled, len(classes))
    
    objects, labels, probs = rcnn_proposals(
            proposals, bbox_pred, cls_prob, image.shape[1:3], 80,
            min_prob_threshold=0.0,
)

objects = objects.numpy()
labels = labels.numpy()
probs = probs.numpy()
