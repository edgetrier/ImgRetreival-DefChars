import os, shutil, logging

if not os.path.isdir('./images'):
    os.mkdir('./images')
    logging.warning('Creating "images" folder - {0}'.format(os.path.abspath('./images')))

if not os.path.isdir('./.images_temp'):
    os.mkdir('./.images_temp')
    logging.warning('Creating "images" folder - {0}'.format(os.path.abspath('./.images_temp')))
else:
    shutil.rmtree('./.images_temp')
    os.mkdir('./.images_temp')
    logging.warning('Initialise "temp" folder - {0}'.format(os.path.abspath('./.images_temp')))

