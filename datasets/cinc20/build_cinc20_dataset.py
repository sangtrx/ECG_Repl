import shutil 
import os


if __name__ == "__main__":

  source = '/content/ECG_Repl/datasets/cinc20/Training_2/'
  dest = '/content/ECG_Repl/datasets/cinc20/WFDB/'

  files = os.listdir(source)
  for f in files:
      shutil.move(source+f, dest)
  shutil.rmtree(source)

  source = '/content/ECG_Repl/datasets/cinc20/Training_PTB/'
  files = os.listdir(source)
  for f in files:
      shutil.move(source+f, dest)
  shutil.rmtree(source)

  source = '/content/ECG_Repl/datasets/cinc20/Training_StPetersburg/'
  files = os.listdir(source)
  for f in files:
      shutil.move(source+f, dest)
  shutil.rmtree(source)

  source = '/content/ECG_Repl/datasets/cinc20/Training_WFDB/'
  files = os.listdir(source)
  for f in files:
      shutil.move(source+f, dest)
  shutil.rmtree(source)

  