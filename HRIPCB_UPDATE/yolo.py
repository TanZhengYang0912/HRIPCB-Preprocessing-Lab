from pylabel import importer
PATH_TO_ANNOTATIONS = "D:\\Output\\train\\annotations"
dataset = importer.ImportVOC(path=PATH_TO_ANNOTATIONS)
dataset.export.ExportToYoloV5()