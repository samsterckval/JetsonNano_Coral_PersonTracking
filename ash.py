# MIT License
# Copyright (c) 2019 Sam Sterckval
# See license
# Class for Subject of interest, that can track centroid with the aid of a siamese classifier

import numpy as np
import cv2
from scipy.spatial import distance as dist
from tftrt_helper import FrozenGraph, TfEngine, TftrtEngine

class Ash():
    '''
    Ash class
    name : display name
    color : display color
    loc : centroid of box
    size : [W,H] size of box
    p1 : upper left point of box
    p2 : lower right piont of box
    area : area of box
    model : 1 leg of the siamese model used to re-ID
    reid_refresh : updates between the refresh of the re-ID, defaults to 5
    mbed_refresh : updates between the refresh of the reference embedding, defaults to 50
    '''
    def __init__(self, color=None, name=None, model=None, reid_refresh=5, mbed_refresh=50):
        if name:
            self.name = name
        else:
            self.name = 'Ash'

        if color:
            self.color = color
        else:
            self.color = [0,255,0]

        if model:
            self.reid = 1
            frozenmodel = FrozenGraph(model, (224, 224, 3))
            print('FrozenGraph build.')
            self.siamese_leg = TftrtEngine(frozenmodel, 1, 'fp16', output_shape=(1024))
            print('TF-TRT model ready to rumble!')
        else:
            self.reid = 0

        self.reid_refresh = reid_refresh
        self.reid_counter = 0
        self.reid_score = 0
        self.mbed_refresh = mbed_refresh
        self.mbed_counter = 0
        self.loc = np.array([0,0])
        self.size = np.array([0,0])
        self.p1 = np.array([0,0])
        self.p2 = np.array([0,0])
        self.ref_mbed = np.zeros((1,1024), dtype='float32')
        self.area = 0
        self.speed = np.array([0,0])
        self.absent = 0
        self.ashed = 0
        self.ma_alpha = 0.3

    def Set_Ref_Mbed(self, frame):
        '''
        Set the reference embedding for the Subject of Interest ( Ash )
        :param img: cropped img SOI
        :return:
        '''
        if self.ashed and self.reid:
            self.ref_mbed = self.siamese_leg.infer(self.Get_Siamese_Tensor(frame))
            print('Reference embedding updated')
        else:
            print('No SOI set yet.')


    def Calc_Area(self):
        '''
        Calculates the area of box
        :return: Nothing
        '''
        self.area = self.size[0] * self.size[1]

    def Calc_Centroid(self):
        '''
        Calculates centroid of box
        '''
        self.loc[0] = int((self.p1[0] + self.p2[0])) >> 1
        self.loc[1] = int((self.p1[1] + self.p2[1])) >> 1

    def Calc_Size(self):
        '''
        Size as np.array([W,H])
        '''
        self.size[0] = self.p2[0] - self.p1[0]
        self.size[1] = self.p2[1] - self.p1[1]

    def Calc_Speed(self, next_box, t):
        '''
        Calculate the instantanous speed
        :param next_box: new bbox [p1x, p1y, p2x, p2y]
        :param t: ms since last frame
        :return: Nothing
        '''
        cX = int((next_box[0] + next_box[2])) >> 1
        cY = int((next_box[1] + next_box[3])) >> 1
        self.speed[0] = (cX - self.loc[0]) / t
        self.speed[1] = (cY - self.loc[1]) / t


    def Get_Closest_Box(self, point, box_list):
        '''
        Get the bbox closest to point from list
        :param point: target point
        :param box_list: list of candidate boxes
        :return: np.array([p1x,p1y,p2x,p2y])
        '''

        plist = []
        for box in box_list:
            cX = int((box[0] + box[2])) >> 1
            cY = int((box[1] + box[3])) >> 1
            plist.append(np.array([cX, cY]))

        D = dist.cdist(np.expand_dims(point, axis=0), plist)[0]
        min_arg = D.argmin()

        # if D[min_arg] > 90.0:
        #     print(D[min_arg])
        #     print('Nothing was really close, but I still took the closest one...')

        new_rect = box_list[min_arg]

        return np.asarray(new_rect), D[min_arg]

    def Get_Siamese_Tensor(self, frame, bbox=None):
        '''
        Gets you the input tensor for the siamese model.
        Crops the image to the self bbox, and expands dims
        :param frame: current frame
        :return: (1,224,224,3) tensor
        '''
        if bbox is None:    # if no bbox was given, take the self bbox
            p1 = self.Get_p1_tuple()
            p2 = self.Get_p2_tuple()
        else:
            p1 = bbox[:2].astype(int)
            p2 = bbox[2:].astype(int)

        croppedframe = frame[p1[1]:p2[1], p1[0]:p2[0], :]
        tensorframe = cv2.resize(croppedframe, (224, 224))  # resize to model input size - linear
        tensorframe = np.expand_dims(cv2.cvtColor(tensorframe, cv2.COLOR_BGR2RGB), axis=0)  # swap channels + expand dims
        return tensorframe

    def Asher(self, candis, img_width, img_height):
        '''
        Lock onto the person closest to the middle of the screen
        :param candis: list of person bbox's
        :param img_width: width of image
        :param img_height: height of image
        :return: Nothing
        '''

        midpoint = np.array([img_width >> 1, img_height >> 1])
        box, _ = self.Get_Closest_Box(midpoint, candis)

        self.p1 = box[:2]
        self.p2 = box[2:]
        self.Calc_Size()
        self.Calc_Area()
        self.Calc_Centroid()
        self.ashed = 1

    def Define_From_Reid(self, boxes, frame, threshold=0.9):
        if len(boxes) == 0:
            return False

        score_list = []
        for box in boxes:
            score = self.Get_Reid_score(self.Get_Siamese_Tensor(frame, box))
            score_list.append(score)

        score_list = np.asarray(score_list)
        # now we should either have a list with the scores
        a = score_list.argmax()  # take out highest score
        if score_list[a] > threshold:
            self.p1 = boxes[a][:2]
            self.p2 = boxes[a][2:]
            self.Calc_Size()
            self.Calc_Area()
            self.Calc_Centroid()
            self.ashed = 1
            self.absent = 0
            self.reid_counter = 0
            self.reid_score = score_list[a]
            return True
        else:
            return False

    def Update(self, candis, t, frame):
        '''
        Update the object
        :param candis: list of candidates
        :param t: ms since last frame
        :param frame: frame
        :return: True if found, False if not
        '''
        if self.absent>= 15:
            self.ashed = 0
            self.absent = 0
            print('SOI deregistered')
            return False

        if not self.ashed:
            if self.reid:
                found = self.Define_From_Reid(candis, frame, threshold=0.96)
                if found:
                    print('SOI has re-entered')
                    return True
                else:
                    return False
            else:
                return False

        self.mbed_counter += 1  # Update the counters
        self.reid_counter += 1  # Because the siamese re-ID network should prob not run every update

        if len(candis) == 0:        # no candidates, no SOI!
            print('SOI not found, no bboxs')
            self.absent += 1
            return False

        next_box, distance = self.Get_Closest_Box(self.loc, candis) # Get the closest bbox, with its distance

        if distance > 150.0 and self.reid:                                        # Big jump, check all the others
            print('That was a big jump, lets check the other boxes')
            found = self.Define_From_Reid(candis, frame, threshold=0.94)
            if found:
                print('Found a better box [' + str(self.reid_score) + '], targetting that one')
                return True
            else:
                print('No box with high re-ID score was found, assuming SOI was absent.')
                self.absent += 1
                return False
        else:           # Small jump, assume it's right
            next_box[:2] = self.ma_alpha * next_box[:2] + (1 - self.ma_alpha) * self.p1 # A bit of smoothing
            next_box[2:] = self.ma_alpha * next_box[2:] + (1 - self.ma_alpha) * self.p2

            self.p1 = next_box[:2]  # Set the new box
            self.p2 = next_box[2:]

            if self.reid_counter >= self.reid_refresh and self.reid:
                self.reid_counter = 0  # reset counter
                self.reid_score = self.Get_Reid_score(self.Get_Siamese_Tensor(frame))  # get the score

                if self.reid_score < 0.93:  # might indicate a person switch
                    print('Re-ID score was low : ' + str(self.reid_score) + ', checking ' + str(
                        len(candis)) + ' boxes for a higher score')
                    found = self.Define_From_Reid(candis, frame, threshold=0.95)
                    if found:
                        print('Better box found [' + str(self.reid_score) + '], new target')
                        self.absent = 0
                    else:
                        print('Nothing found, keeping current, but absenting SOI')
                        self.absent += 1
                        return False
                else:
                    self.absent = 0

            self.Calc_Size()
            self.Calc_Area()
            self.Calc_Centroid()

            if self.mbed_counter >= self.mbed_refresh and self.reid:
                if self.reid_counter > 0:   # also update the re-ID score if it's old news
                    self.reid_counter = 0   # reset that counter
                    self.reid_score = self.Get_Reid_score(self.Get_Siamese_Tensor(frame))   # Calculate score
                if self.reid_score > 0.95:  # don't update if the score is not too certain...
                    self.mbed_counter = 0   # reset counter
                    self.Set_Ref_Mbed(frame)    # Set new reference embedding
                else:
                    print('Re-ID score was too low, no new reference embedding set.')
                    self.mbed_counter = int(self.mbed_refresh/2)

        return True


    def UpdateV0(self, candis, t, frame):   # Old version of the update, this was mainly ducktapy shit
        '''
        Update the object
        :param candis: list of candidates
        :param t: ms since last frame
        :return: True if found, False if not
        '''
        if self.absent >= 10:
            self.ashed = 0
            self.absent = 1
            print('SOI deregistered')

        if not self.ashed and self.reid and not (self.ref_mbed == np.zeros((1,1024), dtype='float32')).all() and len(candis) > 0:
            found = self.Define_From_Reid(candis, frame, threshold=0.97)
            if found :
                self.ashed = 1
            else:
                return False

        if self.ashed == 0: # No SOI defined yet, so no update required
            return False

        self.mbed_counter += 1      # update the counters
        self.reid_counter += 1      # Cause the siamese re-ID network should not run every update

        if len(candis) == 0:        # no candidates, no SOI!
            print('SOI not found, no bboxs')
            self.absent += 1
            return False

        next_box, distance = self.Get_Closest_Box(self.loc, candis)

        if distance > 150.0:        # Here we could maybe also define from re-ID, this might indicate a person switch
            print('That was a big jump...')

        next_box[:2] = self.ma_alpha * next_box[:2] + (1 - self.ma_alpha) * self.p1
        next_box[2:] = self.ma_alpha * next_box[2:] + (1 - self.ma_alpha) * self.p2

        #self.Calc_Speed(next_box, t)   # Speed is not used atm
        self.p1 = next_box[:2]
        self.p2 = next_box[2:]

        if self.reid_counter >= self.reid_refresh and self.reid:
            self.reid_counter = 0   # reset counter
            self.reid_score = self.Get_Reid_score(self.Get_Siamese_Tensor(frame))   # get the score

            if self.reid_score < 0.93:      # might indicate a person switch
                print('Re-ID score was low : ' + str(self.reid_score) + ', checking ' + str(len(candis)) + ' boxes for a higher score')
                found = self.Define_From_Reid(candis, frame, threshold=0.9)
                if not found:
                    self.absent += 1
                    return False


        if self.mbed_counter >= self.mbed_refresh and self.reid:
            if self.reid_score > 0.95:      # don't update if their might be a person switch
                self.mbed_counter = 0
                self.reid_counter = 0
                self.Set_Ref_Mbed(frame)
            else:
                print('Re-ID score was too low, no new reference embedding set.')

        self.Calc_Size()
        self.Calc_Area()
        self.Calc_Centroid()
        self.absent = 0
        return True

    def Get_Reid_score(self, tensor):
        new_embed = self.siamese_leg.infer(tensor)
        return (1 - dist.cosine(self.ref_mbed, np.transpose(new_embed)))

    def Get_p1_tuple(self):
        return (int(self.p1[0]), int(self.p1[1]))

    def Get_p2_tuple(self):
        return (int(self.p2[0]), int(self.p2[1]))

    def Get_centroid_tuple(self):
        return (int(self.loc[0]), int(self.loc[1]))
