import pillowfight
from PIL import Image
import pyocr.builders
from core.imif_digits import *


cascfacePath = "haarcascade_frontalface_default.xml"
cascprofilePath = "haarcascade_profileface.xml"
cascfullbodyPath = "haarcascade_fullbody.xml"
casclowerbodyPath = "haarcascade_lowerbody.xml"

faceCascade = cv2.CascadeClassifier(cascfacePath)
profileCascade = cv2.CascadeClassifier(cascprofilePath)
fullbodyCascade = cv2.CascadeClassifier(cascfullbodyPath)
lowerbodyCascade = cv2.CascadeClassifier(casclowerbodyPath)

imif = imif_digits()
imif.load_model('trained_models/mnist_digits.ckpt')

tools = pyocr.get_available_tools()
tool = tools[0]
langs = tool.get_available_languages()
lang = langs[1]

font = cv2.FONT_HERSHEY_DUPLEX
fontScale = 3
fontColor = (0, 0, 0)
fontColor2 = (0, 128, 255)
thickness = 10
lineType = 2
bottomLeftOrigin = False


def plate_symbol_detected(plate, pre):

    output_plate_symbols = str()

    # расширение области распозноваемого символа
    symbol_area_extension = int(30)

    # ресайз плейта для распознования символов с помошью  pytesseract
    # при возврате к этому методу ресайз нужно сделать итерационным процесом
    '''scale_percent = 80  # для первых двух 61
    width = int(plate.shape[1] * scale_percent / 100)
    height = int(plate.shape[0] * scale_percent / 100)
    dim = (width, height)
    plate = cv2.resize(plate, dim, interpolation=cv2.INTER_NEAREST)'''
    # INTER_NEAREST INTER_LINEAR INTER_AREA INTER_CUBIC INTER_LANCZOS4

    # обработка сырой картинки
    if pre is False:
        plate_grey = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('plate.png', plate_grey)
        # cv2.waitKey(0)

        # plate_grey = cv2.bilateralFilter(plate_grey, 9, 90, 16)
        # blurred = cv2.medianBlur(plate_grey, 5)
        threshold_plate = cv2.threshold(plate_grey, 80, 240, cv2.THRESH_BINARY)[1]
        # threshold_plate = cv2.medianBlur(threshold_plate, 5)
        threshold_plate_inv = cv2.threshold(plate_grey, 80, 240, cv2.THRESH_BINARY_INV)[1]
        # threshold_plate_inv = cv2.medianBlur(threshold_plate_inv, 5)

        # plate_blur = cv2.cv2.GaussianBlur(plate_grey, (3, 3), 0)
        '''plate_blur = cv2.medianBlur(plate_grey, 5)
        ret2, threshold_plate_inv = cv2.threshold(plate_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret3, threshold_plate = cv2.threshold(plate_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)'''
        # threshold_plate_inv = plate_grey

    # когда на вход приходит уже предобработанная картинка
    elif pre is True:
        threshold_plate_inv = cv2.bitwise_not(plate)
        threshold_plate = plate

    ## распознование всего плейта с помошью tesseract
    '''plate_png = Image.fromarray(threshold_plate_inv)
    cv2.imshow("threshold_plate_inv", threshold_plate_inv)
    cv2.waitKey(0)
    digits = tool.image_to_string(
        plate_png,
        lang=lang,
        builder=pyocr.tesseract.DigitBuilder())
    clean_digits = ''
    if len(digits) > 0:
        for s in digits.split():
            if s.isdigit():
                clean_digits = str(s)
                if len(clean_digits) == len(digits):
                    return clean_digits
    return None'''

    ## поиск и обработка символов на плейте
    arr = np.zeros([threshold_plate.shape[0], threshold_plate.shape[1]])
    # threshold_plate.shape [0] -> высота [1] -> ширина
    contours, hierarchy = cv2.findContours(threshold_plate, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contour_dict = dict()

    # сортировка контуров слева направо
    for contour in contours:

        [int_x, int_y, int_width, int_height] = cv2.boundingRect(contour)
        zero_contour = np.zeros((contour.shape[0], contour.shape[1], contour.shape[2]))
        for i in range(len(contour)):
            zero_contour[i] = contour[i] - [[int_x-symbol_area_extension/2, int_y-symbol_area_extension/2]]
        zero_contour = zero_contour.astype(int)

        if int_width < threshold_plate.shape[1] / 2 and int_height >= threshold_plate.shape[0] * 0.6:
            contour_dict[int_x] = [int_x, int_y, int_width, int_height, zero_contour]  # ,contour
            # отрисовка внешних контуров на плейте
            # arr = cv2.drawContours(arr, contour, -1, 255)
            # cv2.imshow("symbol_contour", arr)
            # cv2.waitKey(0)
    sorted_contour_list = list(contour_dict.keys())
    sorted_contour_list.sort()

    # поиск внутренних контуров в сортированных внешних контурах
    for sorted_contour in sorted_contour_list:
        symbol_arr = np.zeros([contour_dict[sorted_contour][3] + symbol_area_extension,
                               contour_dict[sorted_contour][2] + symbol_area_extension])
        symbol_on_plate = threshold_plate_inv[int(contour_dict[sorted_contour][1])-1:
                                 int(contour_dict[sorted_contour][1] + contour_dict[sorted_contour][3]+1),
                                 int(contour_dict[sorted_contour][0]-1):
                                 int(contour_dict[sorted_contour][0] + contour_dict[sorted_contour][2])+1]
        # cv2.imshow("symbol_on_plate", symbol_on_plate)
        # cv2.waitKey(0)

        internal_contours = cv2.findContours(symbol_on_plate, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]

        internal_contours_list = list()
        for internal_contour in internal_contours:
            [x_internal, y_internal, width_internal, height_internal] = cv2.boundingRect(internal_contour)
            if contour_dict[sorted_contour][2] * 0.1 < width_internal < contour_dict[sorted_contour][2] * 0.9:
                # and contour_dict[sorted_contour][3] < height_internal < contour_dict[sorted_contour][3]
                zero_internal = np.zeros((internal_contour.shape[0],
                                          internal_contour.shape[1],
                                          internal_contour.shape[2]))
                for i in range(len(internal_contour)):
                    zero_internal[i] = internal_contour[i] + \
                                       [[int(symbol_area_extension/2), int(symbol_area_extension/2)]]  # - [[x_internal, y_internal]]
                zero_internal = zero_internal.astype(int)
                internal_contours_list.append(zero_internal)
                # отрисовка внутренних контуров
                # arrr = np.zeros([height_internal + symbol_area_extension*2, width_internal + symbol_area_extension*2])
                # arrr = cv2.drawContours(arrr, zero_internal, -1, 255)
                # cv2.imshow("zero_internal", arrr)
                # cv2.waitKey(0)

        # обьединение внешнего контура с его внутренними контурами
        cv2.fillPoly(symbol_arr, pts=[contour_dict[sorted_contour][4]], color=(255, 255, 255))
        for con in internal_contours_list:
            cv2.fillPoly(symbol_arr, pts=[con], color=(0, 0, 0))

        # отрисовка итогового символа на расширенной области
        # cv2.imshow('test_symbol.png', symbol_arr)
        # cv2.waitKey(0)

        # распознвание символа с помошью tesseract (изначально применялось для распознования всего плейта)
        '''symbol_png = Image.fromarray(symbol_arr)
        symbols = tool.image_to_string(
            symbol_png,
            lang=lang,
            builder=pyocr.tesseract.DigitBuilder())
        print('symbol', symbols)'''

        # распознование с помошью обученной сверточной сети
        id_digit = imif.identify(symbol_arr)
        if id_digit != '':
            output_plate_symbols += str(id_digit)

        # cv2.imshow("symbol", symbol)
        # cv2.waitKey(0)
    if len(output_plate_symbols) > 0:
        return output_plate_symbols
    return None

def plate_detected(area, mode, h, w, pre):

    output_list = list()

    # очистка по предположительному размеру плейта
    # сейчас идет расчет относительно размера найденой фичи (лицо, рост, профиль)
    # возможно лучше сделать относительно предпологаемой области нахождения плейта
    '''if mode == 'profiles' or mode == 'faces':
        # параметры для расчета относительно размера фичи
        min_expected_width = h / 4
        min_expected_height = h / 4
        max_expected_width = h * 1.5
        max_expected_height = h * 1.5
        
    elif mode == 'fullbodys':
        min_expected_width = h
        min_expected_height = h
        max_expected_width = h
        max_expected_height = h'''

    # относительно рамера предпологаемой области плейта
    # одинаковые значения для всех фичь, так как размер предпологаемой области почти одинаков
    min_expected_width = int(w / 6)
    min_expected_height = int(h / 15)
    max_expected_width = int(w)
    max_expected_height = int(h / 3)

    # cv2.imshow('area', area)
    # cv2.waitKey(0)
    new = Image.fromarray(area)

    img_out = pillowfight.swt(new, output_type=pillowfight.SWT_OUTPUT_ORIGINAL_BOXES)
    # img_out.show()
    img_out = np.array(img_out)
    img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('img_out', img_out)
    # cv2.waitKey(0)

    # черные блоки областей возвращаемх pillowfight.swt
    area_plate_block = cv2.threshold(img_out, 254, 255, cv2.THRESH_BINARY_INV)[1]
    # cv2.imshow('black_box', area_plate_block)
    # cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(area_plate_block, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # отрисовка контуров черных блоков из pillowfight.swt
    # arr = np.zeros([area.shape[0], area.shape[1]])
    # arr = cv2.drawContours(arr, contours, -1, 255)
    # cv2.imshow('black_box_contour', arr)
    # cv2.waitKey(0)

    # структура contour[0][0][0] : первый элемент - индес точки(против чавовой начиная с левой верхней)
    # второй - всегда 0, третий - 0->x 1->y
    for contour in contours:
        if len(contour) == 4:
            width = contour[3][0][0] - contour[0][0][0]
            height = contour[1][0][1] - contour[0][0][1]

            if (min_expected_width < width < max_expected_width)\
                    and (min_expected_height < height < max_expected_height):
                y1 = contour[0][0][1]
                y2 = contour[1][0][1]
                x1 = contour[0][0][0]
                x2 = contour[3][0][0]

                width_extension = 12
                height_extension = 8

                if y1 - height_extension < 0:
                    y1 = 0
                else:
                    y1 = y1 - height_extension

                if y2 + height_extension > area.shape[0]:
                    y2 = area.shape[0]
                else:
                    y2 = y2 + height_extension

                if x1 - width_extension < 0:
                    x1 = 0
                else:
                    x1 = x1 - width_extension

                if x2 + width_extension > area.shape[1]:
                    x2 = area.shape[1]
                else:
                    x2 = x2 + width_extension

                expected_plate = area[y1:y2, x1: x2]

                output_list.append(plate_symbol_detected(expected_plate, pre))

    clear_output_list = [x for x in output_list if x is not None]
    if len(clear_output_list) > 0:
        print(clear_output_list)
        return clear_output_list
    else:
        return None


def area_detected(image, image_info, mode, x, y, w, h):
    if mode == 'profiles':
        cv2.rectangle(image_info, (x, y), (x + w, y + h), (0, 255, 255), 18)
        area = image[int(y + 1.5 * h):int(y + 5 * h), int(x - 0.5 * h): int(x + 1.5 * w)]
        width = int(x + 1.5 * w) - int(x - 0.5 * h)
        height = int(y + 5 * h) - int(y + 1.5 * h)
        output = plate_detected(area, mode, height, width, pre=False)
        if output is not None:
            cv2.rectangle(image_info, (int(x - 0.5 * h), int(y + 1.5 * h)),
                          (int(x + 1.5 * w), int(y + 5 * h)), (0, 0, 255), 10)
            cv2.putText(image_info, output[0], (int(x - 0.5 * h), int(y + 1.5 * h)),
                        font, fontScale, fontColor, thickness, lineType, bottomLeftOrigin)
        else:
            area_gray = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(area_gray, (5, 5), 0)
            area_preprocessed = cv2.threshold(blurred, 80, 240, cv2.THRESH_BINARY)[1]
            output = plate_detected(area_preprocessed, mode, height, width, pre=True)
            if output is not None:
                cv2.rectangle(image_info, (int(x - 0.5 * h), int(y + 1.5 * h)),
                              (int(x + 1.5 * w), int(y + 5 * h)), (0, 0, 255), 10)
                cv2.putText(image_info, output[0], (int(x - 0.5 * h), int(y + 1.5 * h)),
                            font, fontScale, fontColor, thickness, lineType, bottomLeftOrigin)

    elif mode == 'fullbodys':
        cv2.rectangle(image_info, (x, y), (x + w, y + h), (255, 0, 0), 15)
        area = image[int(y + 0.15 * h):int(y + 0.55 * h), int(x + 0.25 * w): int(x + 0.75 * w)]
        width = int(x + 0.75 * w) - int(x + 0.25 * w)
        height = int(y + 0.55 * h) - int(y + 0.15 * h)
        output = plate_detected(area, mode, height, width, pre=False)
        if output is not None:
            cv2.rectangle(image_info, (int(x + 0.25 * w), int(y + 0.15 * h)),
                          (int(x + 0.75 * w), int(y + 0.6 * h)), (0, 0, 255), 10)
            cv2.putText(image_info, output[0], (int(x + 0.25 * w), int(y + 0.15 * h)),
                        font, fontScale, fontColor, thickness, lineType, bottomLeftOrigin)
        else:
            area_gray = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(area_gray, (5, 5), 0)
            area_preprocessed = cv2.threshold(blurred, 80, 240, cv2.THRESH_BINARY)[1]
            output = plate_detected(area_preprocessed, mode, height, width, pre=True)

            if output is not None:
                cv2.rectangle(image_info, (int(x + 0.25 * w), int(y + 0.15 * h)),
                              (int(x + 0.75 * w), int(y + 0.6 * h)), (0, 0, 255), 10)
                cv2.putText(image_info, output[0], (int(x + 0.25 * w), int(y + 0.15 * h)),
                            font, fontScale, fontColor, thickness, lineType, bottomLeftOrigin)

    elif mode == 'faces':
        cv2.rectangle(image_info, (x, y), (x + w, y + h), (0, 255, 0), 15)
        area = image[int(y + 1.5 * h):int(y + 5 * h), int(x - 0.5 * h): int(x + 1.5 * w)]
        width = int(x + 1.5 * w) - int(x - 0.5 * h)
        height = int(y + 5 * h) - int(y + 1.5 * h)
        output = plate_detected(area, mode, height, width, pre=False)
        if output is not None:
            cv2.rectangle(image_info, (int(x - 0.5 * h), int(y + 1.5 * h)),
                          (int(x + 1.5 * w), int(y + 5 * h)), (0, 0, 255), 10)
            cv2.putText(image_info, output[0], (int(x - 0.5 * h), int(y + 1.5 * h)),
                        font, fontScale, fontColor, thickness, lineType, bottomLeftOrigin)
        else:
            area_gray = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(area_gray, (5, 5), 0)
            area_preprocessed = cv2.threshold(blurred, 80, 240, cv2.THRESH_BINARY)[1]
            output = plate_detected(area_preprocessed, mode, height, width, pre=True)
            if output is not None:
                cv2.rectangle(image_info, (int(x - 0.5 * h), int(y + 1.5 * h)),
                              (int(x + 1.5 * w), int(y + 5 * h)), (0, 0, 255), 10)
                cv2.putText(image_info, output[0], (int(x - 0.5 * h), int(y + 1.5 * h)),
                            font, fontScale, fontColor, thickness, lineType, bottomLeftOrigin)


def image_detect(imagepath, fa_sc, fa_neig, fa_min, fa_max,
                 pr_sc, pr_neig, pr_min, pr_max,
                 fu_sc, fu_neig, fu_min, fu_max):
    image = cv2.imread(imagepath)
    cv2.namedWindow("original_image image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("original_image image", 900, 600)
    cv2.imshow("original_image image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    image_info = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=fa_sc,
        minNeighbors=fa_neig,
        minSize=fa_min,
        maxSize=fa_max,
        flags=cv2.CASCADE_SCALE_IMAGE)  # cv2.CASCADE_SCALE_IMAGE || cv2.cv.CV_HAAR_SCALE_IMAGE

    profiles = profileCascade.detectMultiScale(
        gray,
        scaleFactor=pr_sc,
        minNeighbors=pr_neig,
        minSize=pr_min,
        maxSize=pr_max,
        flags=cv2.CASCADE_SCALE_IMAGE)  # cv2.CASCADE_SCALE_IMAGE || cv2.cv.CV_HAAR_SCALE_IMAGE

    fullbodys = fullbodyCascade.detectMultiScale(
        gray,
        scaleFactor=fu_sc,
        minNeighbors=fu_neig,
        minSize=fu_min,
        maxSize=fu_max,
        flags=cv2.CASCADE_SCALE_IMAGE)  # cv2.CASCADE_SCALE_IMAGE || cv2.cv.CV_HAAR_SCALE_IMAGE

    # print("Found {0} faces!".format(len(faces)))

    for (x, y, w, h) in faces:
        mode = 'faces'
        area_detected(image, image_info, mode, x, y, w, h)

    for (x, y, w, h) in fullbodys:
        mode = 'fullbodys'
        area_detected(image, image_info, mode, x, y, w, h)

    for (x, y, w, h) in profiles:
        mode = 'profiles'
        area_detected(image, image_info, mode, x, y, w, h)

    cv2.namedWindow("output image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("output image", 900, 600)
    cv2.imshow("output image", image_info)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('test1_area_face_recognition.jpg', image_info)


# ===============================================================================================================
# ===============================================================================================================


image_detect(imagepath="test1f.jpg", fa_sc=1.15, fa_neig=15, fa_min=(80, 80), fa_max=(2000, 2000),
             pr_sc=1.1, pr_neig=15, pr_min=(80, 80), pr_max=(400, 400),
             fu_sc=1.1, fu_neig=5, fu_min=(100, 100), fu_max=(2000, 2000), )

