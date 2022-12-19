import numpy as np
import cv2
import plotly.express as px
import matplotlib.pyplot as plt


img = cv2.imread('slika2.png', 0)
#img = cv2.resize(img,(550,500),interpolation=cv2.INTER_AREA)
mean, std = cv2.meanStdDev(img)
original = img.copy()

#BINARIZATION
k1 = 0.65
k2 = 0.8
t = np.uint8(k1*std + k2*mean)
img = cv2.threshold(img, t[0,0], 255, cv2.THRESH_BINARY)[1]

#CONNECTED COMPONENT LABELING
num_labels, labels = cv2.connectedComponents(img) #num_labels-broj povezanih komp, labels-razliciti brojevi su razl povezane komp, 0 je pozadina
labels_row = labels.ravel()
labels_count = np.bincount(labels_row) #koliko se koji broj pojavljuje
labels_sort = np.sort(labels_count)[::-1] #sortira u opadajucem redosledu
num = labels_sort[1] #najvise puta se pojavljuje broj koji predstavlja pozadinu, a drugi po redu najvise je onaj koji predstavlja lice
idx = np.where(labels_count == num)  #to je broj koji predstavlja lice

#HOLE FILLING
img2 = np.zeros(img.shape)
for i in range(0,img.shape[0]):
    for j in range(0,img.shape[1]):
        if(labels[i,j]==idx[0][0]):
            img2[i,j] = img[i,j]    #img2 je binarna slika koja ima bele samo piksele koji pripadaju povezanoj komponenti koja predstavlja lice

fig = px.imshow(img2, color_continuous_scale='gray')
#fig.show()
se = np.ones((14,14),np.uint8)#ovo ni ne mora
img_cl = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, se)  #ovo ni ne mora
fig1 = px.imshow(img_cl,color_continuous_scale='gray')
#fig1.show()

#BOUNDING BOX
contours, hierarchy = cv2.findContours(np.uint8(img2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(hierarchy)
for c in contours:
    x, y, w, h = cv2.boundingRect(c) #(x,y) are the top-left coordinate of the rectangle and (w,h) are its width and height.
    cv2.rectangle(original, (x, y), (x + w, y + h), (255,0,0), 2) #na original se iscrtava box, (255,0,0),2 je vezano za boju boxa
    face = original[y:y + h, x:x + w]
    cv2.imwrite("face.png", face)


fig2 = px.imshow(original, color_continuous_scale='gray')
#fig2.show()

#ISECI LICE
img_face = cv2.imread('face.png', 0)
fig3 = px.imshow(img_face, color_continuous_scale='gray')
#fig3.show()

#DETEKCIJA OCIJU, USTA I NOSA
#MASKING

mask1 = np.zeros(img_face.shape)
mask1[np.int(img_face.shape[0]/10*3):np.int(img_face.shape[0]*5/9),5:(img_face.shape[1]-5)] = 255
fig4 = px.imshow(mask1, color_continuous_scale='gray')
#fig4.show()

mask2 = np.zeros(img_face.shape)
mask2[np.int(img_face.shape[0]*5/9):(img_face.shape[0]-5),5:(img_face.shape[1]-5)] = 255
#fig5 = px.imshow(mask2, color_continuous_scale='gray')
#fig5.show()

k11 = 0.65
k22 = 0.5
t1 = np.uint8(k11*std + k22*mean)
se1 = np.ones((3,3),np.uint8)
se2 = np.ones((15,15),np.uint8)

img_mask1 = img_face.copy()
img_mask1[mask1 == 0] = 0
img_mask1[mask1 != 0] = img_face[mask1 != 0]
part1 = cv2.threshold(img_mask1, t[0,0], 255, cv2.THRESH_BINARY)[1]
part1_cl = cv2.morphologyEx(part1, cv2.MORPH_OPEN, se1)

img_mask2 = img_face.copy()
img_mask2[mask2 == 0] = 0
img_mask2[mask2 != 0] = img_face[mask2 != 0]
part2 = cv2.threshold(img_mask2, t1[0,0], 255, cv2.THRESH_BINARY)[1]
part2_cl = cv2.morphologyEx(part2, cv2.MORPH_OPEN, se2)

part1_2 = np.zeros(part1.shape)
part2_2 = np.zeros(part2.shape)
for i in range(0,part1.shape[0]):
    for j in range(0, part1.shape[1]):
        if(part1_cl[i,j]==0):
            part1_2[i,j] = 255

for i in range(0,part2.shape[0]):
    for j in range(0, part2.shape[1]):
        if(part2_cl[i,j]==0):
            part2_2[i,j] = 255

fig6 = px.imshow(part1_2, color_continuous_scale='gray')
#fig6.show()
fig7 = px.imshow(part2_2, color_continuous_scale='gray')
#fig7.show()

#LABELING
num_labels1, labels1 = cv2.connectedComponents(np.uint8(part1_2))
num_labels2, labels2 = cv2.connectedComponents(np.uint8(part2_2))

####
label_hue = np.uint8(179*labels2/np.max(labels2))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
labeled_img[label_hue == 0] = 0

# Showing Original Image
plt.imshow(cv2.cvtColor(part2, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Orginal Image")
plt.show()

# Showing Image after Component Labeling
plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Image after Component Labeling")
plt.show()
###
#GORNJI DEO
labels1_row = labels1.ravel()
labels1_count = np.bincount(labels1_row) #koliko se koji broj pojavljuje
labels1_sort = np.sort(labels1_count)[::-1]

num_el1 = len(labels1_sort)
num1 = labels1_sort[2]
idx1 = np.where(labels1_count == num1)

if(num_el1>=4):
    num2 = labels1_sort[3]
    idx2 = np.where(labels1_count == num2)

else:
    idx2=idx1

img_eyes = np.zeros(labels1.shape)
for i in range(0,labels1.shape[0]):
    for j in range(0,labels1.shape[1]):
        if(labels1[i,j]==idx1[0][0] or labels1[i,j]==idx2[0][0]):
            img_eyes[i,j] = part1_2[i,j]

fig8 = px.imshow(img_eyes,color_continuous_scale='gray')
#fig8.show()

contours1, hierarchy1 = cv2.findContours(np.uint8(img_eyes), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(hierarchy1)
w1=[]
h1=[]
x1=[]
y1=[]
i=0
c2 = max(contours1, key = cv2.contourArea)#the biggest countour (c) by the area
c3 = min(contours1, key = cv2.contourArea)
if(len(contours1) == 1):
    c4 = contours1
    w1 = [0 for i in range(1)]
    h1 = [0 for i in range(1)]
    x1 = [0 for i in range(1)]
    y1 = [0 for i in range(1)]
else:
    c4 = (c2, c3)
    w1 = [0 for i in range(2)]
    h1 = [0 for i in range(2)]
    x1 = [0 for i in range(2)]
    y1 = [0 for i in range(2)]

for c1 in c4:
    x1[i], y1[i], w1[i], h1[i] = cv2.boundingRect(c1) #(x,y) are the top-left coordinate of the rectangle and (w,h) are its width and height.
    if(np.array_equal(c1, c2, equal_nan=False)): #isctace box samo ako je on najveci
        cv2.rectangle(img_face, (x1[i], y1[i]), (x1[i] + w1[i], y1[i] + h1[i]), (255,0,0), 2) #na original se iscrtava box, (255,0,0),2 je vezano za boju boxa
    if(i==1):
        if((w1[1]*h1[1])<(w1[0]*h1[0])/2): #ako je jedan box duplo manji od drugog
            if (x1[0] < img_face.shape[1] / 2 - 5):  # detektovano je levo oko
                cv2.rectangle(img_face, (np.int(x1[0] + img_face.shape[1]/2), y1[0]),(np.int(x1[0] + img_face.shape[1]/2 + w1[0]), y1[0] + h1[0]), (255, 0, 0),2)  # simetricno preslikan drugi box
            else:  # detektovano je desno oko
                cv2.rectangle(img_face, (np.int(x1[0] - img_face.shape[1]/2), y1[0]),(np.int(x1[0] - img_face.shape[1]/2 + w1[0]), y1[0] + h1[0]),(255, 0, 0), 2)
        else:
            cv2.rectangle(img_face, (x1[i], y1[i]), (x1[i] + w1[i], y1[i] + h1[i]), (255, 0, 0), 2)
    elif (len(contours1) == 1):  # ako je detektovano samo jedno oko
        if (x1[i] < img_face.shape[1] / 2 - 5):  # detektovano je levo oko
            cv2.rectangle(img_face, (np.int(x1[i] + img_face.shape[1] / 2), y1[i]), (np.int(x1[i] + img_face.shape[1] / 2 + w1[i]), y1[i] + h1[i]), (255, 0, 0),2)  # simetricno preslikan drugi box
        else:  # detektovano je desno oko
            cv2.rectangle(img_face, (np.int(x1[i] - img_face.shape[1] / 2), y1[i]),(np.int(x1[i] - img_face.shape[1] / 2 + w1[i]), y1[i] + h1[i]), (255, 0, 0), 2)
    i=i+1

#fig9 = px.imshow(img_face, color_continuous_scale='gray')
#fig9.show()


#DONJI DEO
labels2_row = labels2.ravel()
labels2_count = np.bincount(labels2_row) #koliko se koji broj pojavljuje
labels2_sort = np.sort(labels2_count)[::-1]

num_el2 = len(labels2_sort)

if(num_el2>=3): #ako ima bar usta ili nos
    num3 = labels2_sort[2]
    idx3 = np.where(labels2_count == num3)
else: #ako nije nista detektovao
    idx3 = np.where(labels2_count == labels2_sort[0])

if(num_el2>=4): #sigurno ima dva elementa a to su lice i pozadina, 3. su usta a 4. nos
    num4 = labels2_sort[3]
    idx4 = np.where(labels2_count == num4)

else:
    idx4 = idx3 #samo usta ili nos detektovao

img_mouth = np.zeros(part2.shape)
for i in range(0,part2.shape[0]):
    for j in range(0,part2.shape[1]):
        if(num_el2>=3): #ako nije detektovao nista onda nece uci ovde tj img_mouth ce biti crna tj len(countors2)=0
            if(labels2[i,j]==idx3[0][0] or labels2[i,j]==idx4[0][0]):
                img_mouth[i,j] = part2_2[i,j]

fig9 = px.imshow(img_mouth,color_continuous_scale='gray')
fig9.show()

contours2, hierarchy2 = cv2.findContours(np.uint8(img_mouth), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours2))
w2=[]
h2=[]
x2=[]
y2=[]
j=0

if(num_el2>=3):
    c5 = max(contours2, key=cv2.contourArea)  # the biggest countour (c) by the area
    c6 = min(contours2, key=cv2.contourArea)
    if (len(contours2) == 1):
        c7 = contours2
        w2 = [0 for j in range(1)]
        h2 = [0 for j in range(1)]
        x2 = [0 for j in range(1)]
        y2 = [0 for j in range(1)]
    else:
        c7 = (c5, c6)
        w2 = [0 for j in range(2)]
        h2 = [0 for j in range(2)]
        x2 = [0 for j in range(2)]
        y2 = [0 for j in range(2)]

    for c8 in c7:
        x2[j], y2[j], w2[j], h2[j] = cv2.boundingRect(c8) #(x,y) are the top-left coordinate of the rectangle and (w,h) are its width and height.
        if (np.array_equal(c8, c5, equal_nan=False)):  # isctace box samo ako je on najveci
            cv2.rectangle(img_face, (x2[j], y2[j]), (x2[j] + w2[j], y2[j] + h2[j]), (255, 0, 0),2)  #na original se iscrtava box, a boju boxa
        if (j == 1):
            if ((w2[1] * h2[1]) < (w2[0] * h2[0])/10):  # nos ili usta su tacka(tacka je 1)
                if (y2[1] < img_face.shape[0] / 4):  # usta su tacka, iscrtava usta
                    cv2.rectangle(img_face, (np.int(img_face.shape[1] * 0.25), np.int(img_face.shape[0] * 0.75)),(np.int(img_face.shape[1] * 0.25 + w1[0]), np.int(img_face.shape[0] * 0.75 + h1[0])),(255, 0, 0), 2)
                else:  #  nos je tacka, iscrvata nos
                    cv2.rectangle(img_face, (np.int(img_face.shape[1] * 0.3), np.int(img_face.shape[0] * 0.6)),(np.int(img_face.shape[1] * 0.3 + w1[0]), np.int(img_face.shape[0] * 0.6 + h1[0])),(255, 0, 0), 2)
                    #cv2.rectangle(img_face, (x2[0]+np.int(w2[0]/4), y2[0]-(h2[0])), (x2[0] +np.int(w2[0]/4)+ np.int(w2[0]/2), y2[0]-(h2[0])+np.int(h2[0]/2)), (255, 0, 0), 2)
            else:
                cv2.rectangle(img_face, (x2[j], y2[j]), (x2[j] + w2[j], y2[j] + h2[j]), (255, 0, 0), 2)
        elif (len(contours2) == 1):  # ako je detektovano samo usta ili nos
            if (y2[j] < img_face.shape[0] / 4):  # detektovano je usta
                cv2.rectangle(img_face, (np.int(img_face.shape[1]*0.3), np.int(img_face.shape[0]*0.6)), (np.int(img_face.shape[1]*0.3+w1[0]), np.int(img_face.shape[0]*0.6+h1[0])), (255, 0, 0), 2)
            else:  # detektovano je nos
                cv2.rectangle(img_face, (np.int(img_face.shape[1]*0.25), np.int(img_face.shape[0]*0.75)), (np.int(img_face.shape[1]*0.25+w1[0]), np.int(img_face.shape[0]*0.75+h1[0])), (255, 0, 0), 2)
        j=j+1
else: #nije nista detektovao
    cv2.rectangle(img_face, (np.int(img_face.shape[1]*0.17), np.int(img_face.shape[0]*0.7)), (np.int(img_face.shape[1]*0.17+w1[0]*2), np.int(img_face.shape[0]*0.7+h1[0]*2)), (255, 0, 0), 2) #usta
    cv2.rectangle(img_face, (np.int(img_face.shape[1]*0.3), np.int(img_face.shape[0]*0.6)), (np.int(img_face.shape[1]*0.3+w1[0]), np.int(img_face.shape[0]*0.6+h1[0])), (255, 0, 0), 2) #nos

fig10 = px.imshow(img_face, color_continuous_scale='gray')
fig10.show()
