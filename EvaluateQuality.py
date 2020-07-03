# Srikanth
import numpy as np

from keras.models import load_model
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array

from PIL import Image
from matplotlib import pylab as pl
#Do NOT remove this line
from mpl_toolkits.mplot3d.axes3d import Axes3D
#Do NOT remove this line
import tensorflow as tf
from path import Path


TABD = "\t"
CSVD = ","
newld = "\n"
writedelim = CSVD

AEOLIAN = "Label: Aeolian"
DRY = "Label: Dry"
GLACIAL = "Label: Glacial"
VOLCANIC = "Label: Volcanic"

# Setting up the model1
def setModel1():
    base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)
    global modelone
    modelone = Model(base_model.input, x)
    modelone.load_weights('./zmodel/mobilenet_weights.h5')
    modelone.compile('sgd','mse')

# Setting up the model2
def setModel2():
    img_width, img_height = 150, 150
    model_path = './zmodel/model.h5'
    model_weights_path = './zmodel/weights.h5'
    global modeltwo
    modeltwo = load_model(model_path)
    modeltwo.load_weights(model_weights_path)
    modeltwo.compile('sgd','mse')

# Truncate
def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

# calculate mean score for AVA dataset
def mean_score(scores):
    si = np.arange(1, 11, 1)
    mean = np.sum(scores * si)
    return mean

# calculate standard deviation of scores for AVA dataset
def std_score(scores):
    si = np.arange(1, 11, 1)
    mean = mean_score(scores)
    std = np.sqrt(np.sum(((si - mean) ** 2) * scores))
    return std



def calculateQuality(threshold):
    print(" Finding Image aesthetic quality ...  ... with threshold : "+ str(threshold))
    print("... ...")

    with tf.device('/CPU:0'):
        img_path1 = 'qualitydata/'
        imgs1 = Path(img_path1).files('*.jpg')

        lesscount = 0
        morecount = 0
        f = open('imagesquality.csv', 'w')

        for img_path1 in imgs1:
            #print("Working on : " +img_path1)
            img = load_img(img_path1)
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)

            x = preprocess_input(x)

            scores = modelone.predict(x, batch_size=1, verbose=1)[0]
            correction = 1.25
            mean = mean_score(scores) + correction
            std = std_score(scores)

            f.write(img_path1)
            f.write(writedelim)
            f.write(str(truncate(mean, 2)))
            f.write(writedelim)
            f.write(str(truncate(std, 2)))
            f.write(newld)

            if mean >= threshold:
                morecount = morecount + 1
            else:
                lesscount = lesscount + 1

            #print(img_path1 + ", NIMA Score : %0.3f +- (%0.3f)" % (mean, std))
        print("Number of images above threshold " + str(morecount))
        print("Number of images below threshold " + str(lesscount))

        f.close()
        print("~~~ Finished writing to the file named imagesquality.csv ~~~")



def classifyImage():
    print(" Classifying Images ...  ...")
    print("... ...")

    aeo_tp = 0
    aeo_tn = 0
    aeo_fp = 0
    aeo_fn = 0

    dry_tp = 0
    dry_tn = 0
    dry_fp = 0
    dry_fn = 0

    gla_tp = 0
    gla_tn = 0
    gla_fp = 0
    gla_fn = 0

    vol_tp = 0
    vol_tn = 0
    vol_fp = 0
    vol_fn = 0
    #
    aeolian_t = 0
    aeolian_f = 0
    dry_t = 0
    dry_f = 0
    glacial_t = 0
    glacial_f = 0
    volcanic_t = 0
    volcanic_f = 0

    with tf.device('/CPU:0'):
        f = open('imagesclassification.csv', 'w')
        img_path2 = 'classdata/'
        imgs2 = Path(img_path2).files('*.jpg')
        img_width, img_height = 150, 150
        count = 1
        count1=0
        count2=0
        count3=0
        count4=0
        counter = 0

        for img_path2 in imgs2:
            counter+=1
            #print(str(img_path2))
            x = load_img(img_path2, target_size=(img_width, img_height))
            x = img_to_array(x)
            x = np.expand_dims(x, axis=0)
            array = modeltwo.predict(x)
            result = array[0]
            answer = np.argmax(result)

            if answer == 0:
                result = AEOLIAN
            elif answer == 1:
                result = DRY
            elif answer == 2:
                result = GLACIAL
            elif answer == 3:
                result = VOLCANIC

            if "aeolian" in str(img_path2):
                count1+=1
            #if count <= 15:
                print(img_path2 + " ..True type : " + AEOLIAN + TABD + "Classification type : " + result)
                if result == AEOLIAN:
                    aeolian_t += 1#
                    aeo_tp += 1
                    dry_tn += 1
                    gla_tn += 1
                    vol_tn += 1

                elif result == DRY:
                    aeolian_f += 1#
                    aeo_fn += 1
                    dry_fp += 1

                    gla_tn += 1
                    vol_tn += 1

                elif result == GLACIAL:
                    aeolian_f += 1#
                    aeo_fn += 1
                    gla_fp+=1

                    dry_tn += 1
                    vol_tn += 1
                elif result == VOLCANIC:
                    aeolian_f += 1#
                    aeo_fn += 1
                    vol_fp += 1

                    dry_tn += 1
                    gla_tn += 1

            elif "dry" in str(img_path2): # img_path2 contains Dry
                count2 += 1
            #elif count > 15 and count <= 30:
                print(img_path2 + " ..True type : " + DRY + TABD + "Classification type : " + result)
                if result == AEOLIAN:
                    dry_f += 1  #
                    dry_fn += 1
                    aeo_fp += 1

                    vol_tn += 1
                    gla_tn += 1

                elif result == DRY:
                    dry_t += 1  #
                    dry_tp += 1
                    aeo_tn += 1
                    gla_tn += 1
                    vol_tn += 1
                elif result == GLACIAL:
                    dry_f += 1  #
                    dry_fn += 1
                    gla_fp += 1

                    aeo_tn += 1
                    vol_tn += 1

                elif result == VOLCANIC:
                    dry_f += 1  #
                    dry_fn += 1
                    vol_fp += 1

                    aeo_tn += 1
                    gla_tn += 1

            elif "glacier" in str(img_path2): # img_path2 contains Glacial
                count3 += 1
            #elif count > 30 and count <= 45:
                print(img_path2 + " ..True type : " + GLACIAL + TABD + "Classification type : " + result)
                if result == AEOLIAN:
                    glacial_f += 1  #
                    gla_fn += 1
                    aeo_fp += 1

                    dry_tn += 1
                    vol_tn += 1

                elif result == DRY:
                    glacial_f += 1  #
                    gla_fn += 1
                    dry_fp += 1

                    aeo_tn += 1
                    vol_tn += 1
                elif result == GLACIAL:
                    glacial_t += 1  #
                    gla_tp += 1
                    aeo_tn += 1
                    dry_tn += 1
                    vol_tn += 1
                elif result == VOLCANIC:
                    glacial_f += 1  #
                    gla_fn += 1
                    vol_fp += 1

                    aeo_tn += 1
                    dry_tn += 1

            elif "volcanic" in str(img_path2): # img_path2 contains Volcanic
            #elif count > 45 and count <= 60:
                count4+=1
                print(img_path2 + " ..True type : " + VOLCANIC + TABD + "Classification type : " + result)
                if result == AEOLIAN:
                    volcanic_f += 1  #
                    vol_fn += 1
                    aeo_fp += 1

                    gla_tn += 1
                    dry_tn += 1

                elif result == DRY:
                    volcanic_f += 1  #
                    vol_fn += 1
                    dry_fp += 1

                    gla_tn += 1
                    aeo_tn += 1

                elif result == GLACIAL:
                    volcanic_f += 1  #
                    vol_fn += 1
                    gla_fp += 1

                    aeo_tn += 1
                    dry_tn += 1
                elif result == VOLCANIC:
                    volcanic_t += 1  #
                    vol_tp += 1
                    aeo_tn += 1
                    dry_tn += 1
                    gla_tn += 1

            count = count + 1

    print(str(counter) +"... ..."+str(count1)+ "... ..."+str(count2)+ "... ..."+str(count3) + "... ..."+str(count4))
    
    print("aeo_tp : " + str(aeo_tp) + "  aeo_tn : " + str(aeo_tn) + "  aeo_fp : " + str(aeo_fp) + "  aeo_fn : " + str(aeo_fn))
    print("dry_tp : " + str(dry_tp) + "  dry_tn : " + str(dry_tn) + "  dry_fp : " + str(dry_fp) + "  dry_fn : " + str(dry_fn))
    print("gla_tp : " + str(gla_tp) + "  gla_tn : " + str(gla_tn) + "  gla_fp : " + str(gla_fp) + "  gla_fn : " + str(gla_fn))
    print("vol_tp : " + str(vol_tp) + "  vol_tn : " + str(vol_tn) + "  vol_fp : " + str(vol_fp) + "  vol_fn : " + str(vol_fn))

    # recall = tp / (tp + fn)
    # precision = tp / (tp + fp)

    aeorec = truncate(aeo_tp / (aeo_tp + aeo_fn) , 2)
    aeoprec = truncate(aeo_tp / (aeo_tp + aeo_fp) , 2)

    dryrec = truncate(dry_tp / (dry_tp + dry_fn), 2)
    dryprec = truncate(dry_tp / (dry_tp + dry_fp), 2)

    glarec = truncate(gla_tp / (gla_tp + gla_fn), 2)
    glaprec = truncate(gla_tp / (gla_tp + gla_fp), 2)

    volrec = truncate(vol_tp / (vol_tp + vol_fn), 2)
    volprec = truncate(vol_tp / (vol_tp + vol_fp), 2)


    print("Precision for Aeolian  : " + str(aeorec) + "\t Recall for Aeolian : " + str(aeoprec))
    print("Precision for Dry  : " + str(dryrec) + "\t Recall for Dry : " + str(dryprec))
    print("Precision for Glacial  : " + str(glarec) + "\t Recall for Glacial : " + str(glaprec))
    print("Precision for Volcanic  : " + str(volrec) + "\t Recall for Volcanic : " + str(volprec))
    print("... ...")

    f.write("Precision for Aeolian  : ")
    f.write(writedelim)
    f.write(str(aeorec))
    f.write(writedelim)
    f.write("Recall for Aeolian  : ")
    f.write(writedelim)
    f.write(str(aeoprec))
    f.write(newld)

    f.write("Precision for Dry  : ")
    f.write(writedelim)
    f.write(str(dryrec))
    f.write(writedelim)
    f.write("Recall for Dry  : ")
    f.write(writedelim)
    f.write(str(dryprec))
    f.write(newld)

    f.write("Precision for Glacial  : ")
    f.write(writedelim)
    f.write(str(glarec))
    f.write(writedelim)
    f.write("Recall for Glacial  : ")
    f.write(writedelim)
    f.write(str(glaprec))
    f.write(newld)

    f.write("Precision for Volcanic  : ")
    f.write(writedelim)
    f.write(str(volrec))
    f.write(writedelim)
    f.write("Recall for Volcanic  : ")
    f.write(writedelim)
    f.write(str(volprec))
    f.write(newld)

    # F1=2TP/2TP+FP+FN
    f.close()
    print("~~~ Finished writing to the file named imagesclassification.csv ~~~")


def convert3Dprojection():
    print(" Converting Images ...  ...")
    print("... ...")

    img_path3 = '2D/'
    # img_path3_output = '3D'
    imgs3 = Path(img_path3).files('*.jpg')

    count = 0
    for img_path3 in imgs3:
        count = count + 1
        print("Working on " + img_path3 + "- " + str(count))

        img = Image.open(img_path3).convert('L')
        z = np.asarray(img)
        mydata = z[::1, ::1]
        fig = pl.figure(facecolor='w')
        ax1 = fig.add_subplot(1, 2, 1)
        im = ax1.imshow(mydata, interpolation='nearest', cmap=pl.cm.jet)
        ax1.set_title('2D')

        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        x, y = np.mgrid[:mydata.shape[0], :mydata.shape[1]]
        ax2.plot_surface(x, y, mydata, cmap=pl.cm.jet, rstride=1, cstride=1, linewidth=0., antialiased=False)
        ax2.set_title('3D')
        ax2.set_zlim3d(0, 100)

        filename = '3D/result' + str(count) + '.png'
        fig.savefig(filename)





def print_menu():  ## Your menu design here
    print(30 * "-", "MENU", 30 * "-")
    print("1. Option 1 : Image aesthetic quality")
    print("2. Option 2 : Image classification")
    print("3. Option 3 : 2D to 3D approximate projection")
    print("4. Exit")
    print(67 * "-")





if __name__ == '__main__':
    setModel1()
    setModel2()

    loop = True

    while loop:  ## While loop which will keep going until loop = False
        print_menu()  ## Displays menu
        choice = input("Enter your choice [1-4]: ")

        if int(choice) == 1:
            print("Menu 1 (Image aesthetic quality) has been selected")
            thresh = input("Enter a threshold between 1 and 10. suggested - 5   :\n")
            threshold = float(thresh)
            if threshold >= 0 and threshold <= 10:
                calculateQuality(threshold)

        elif int(choice) == 2:
            print("Menu 2 (Image classification) has been selected")
            classifyImage()

        elif int(choice) == 3:
            print("Menu 3 (2D to 3D projection) has been selected")
            convert3Dprojection()

        elif int(choice) == 4:
            print("Menu 4 has been selected, exiting the program")
            loop = False

        else:
            # Any integer inputs other than values 1-4 we print an error message
            input("Wrong option selection. Enter any key to try again..")