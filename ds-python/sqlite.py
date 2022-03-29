import sqlite3
from sqlite3 import Error
import pickle
from cifar_10 import *

def unpickle(file):
    with open(file, 'rb') as fo:
        mydict = pickle.load(fo, encoding='bytes')
    return mydict

def create_connection(db_file):
    """
    Creates connection to an SQLite database. 

    :param db_file: name of the SQLite database in string
    :return: sqlite3.Connection to existing or newly created SQLite database
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

def execute_command(conn, sql_command):
    """
    Executes SQL command in a database. 

    :param conn: sqlite3.Connection to target SQLite database
    :param create_table_sql: SQLite command as string
    """
    try:
        c = conn.cursor()
        c.execute(sql_command)
    except Error as e:
        print(e)


def create_image(conn, image):
    """
    Creates a new image row in an images database. 

    :param conn: sqlite3.Connection to an images database
    :param image: (int, string, string) tuple consisting of image label, vector blob file location, and PIL pickle file location
    :return: returns id of the inserted row
    """
    sql_command = """INSERT INTO images(label, resnet, alexnet, pil)
                    VALUES(?,?,?,?)"""
    cur = conn.cursor()
    cur.execute(sql_command, image)
    conn.commit
    print("image inserted", i)
    return cur.lastrowid

def save_as_pickle(obj, filename):
    """
    Converts a python object image into a pickle file. 
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def save_ndarray_as_blob(arr, filename):
    """
    Saves ndarray as binary blob. 
    """
    arr.tofile(filename)

def list_to_sql_image(all_dicts, all_images, all_resnet, all_alexnet, i):
    """
    Creates a single (label, vector blob, pil blob) tuple for the given information. 

    :param all_dicts: list of dictionaries returned by cifar.all_dicts()
    :param all_images: list of PIL images returned by cifar.get_all_pil_images()
    :param all_resnet: list of ResNet vectors returned by cifar.get_all_resnet()
    :param i: index of target image between 0 and 49999
    """
    save_as_pickle(all_images[i], 'tempimg.pickle')
    save_ndarray_as_blob(all_resnet[i], 'tempresnet')
    save_ndarray_as_blob(all_alexnet[i], 'tempalexnet')

    label = all_dicts[i // 10000][b"labels"][i % 10000]
    with open('tempresnet', 'rb') as resfile:
        resnet_blob = resfile.read()
    with open('tempalexnet', 'rb') as alexfile:
        alexnet_blob = alexfile.read()
    with open('tempimg.pickle', 'rb') as imgfile:
       pil_blob = imgfile.read()
    image = (label, resnet_blob, alexnet_blob, pil_blob)
    print("image created", i)
    return image

"""
Obtains PIL images and vectors using cifar.py. 
Then loads it onto a cifar.db SQLite3 database under an "images" table. 
"""
if __name__ == '__main__':

    # Create database, establish connection, add images table
    database = r"cifar.db"
    sql_create_image_table = """CREATE TABLE IF NOT EXISTS images (
                                id INTEGER PRIMARY KEY,
                                label INTEGER NOT NULL,
                                resnet BLOB,
                                alexnet BLOB,
                                pil BLOB
                                ); """
    conn = create_connection(database)
    execute_command(conn, sql_create_image_table)

    # Verify that table has been created
    #cursor = conn.cursor()
    #cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    #print(cursor.fetchall())

    # Obtain labels, PIL images and Resnet vectors
    all_dicts = unpickle_all()
    all_images = get_all_pil_images(all_dicts)
    all_resnet = unpickle("cifar-10-vectors")
    all_alexnet = unpickle("cifar-10-vectors-alexnet")

    # Load the obtained information to SQLite database
    for i in range(50000):
        sql_img = list_to_sql_image(all_dicts, all_images, all_resnet, all_alexnet, i)
        create_image(conn, sql_img)
        conn.commit()
        # Verify that row has been added
        #cursor = conn.cursor()
        #cursor.execute("SELECT * FROM images")
        #print(cursor.fetchall())
    
    conn.commit()
    conn.close()

