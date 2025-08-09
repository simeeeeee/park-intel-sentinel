from MFRC522 import MFRC522
import time
import sys

def uidToString(uid):
    mystring = ""
    for i in uid:
        mystring = format(i, '02X') + mystring
    return mystring

class myRFIDReader(MFRC522):
    def __init__(self, bus=0, dev=0):
        super().__init__(bus=bus, dev=dev)
        self.key = None
        self.keyIn = False
        self.keyValidCount = 0

    def Read(self):
        status, TagType = self.MFRC522_Request(super().PICC_REQIDL)
        if status == self.MI_OK:
            status, uid = self.MFRC522_SelectTagSN()
            if status == self.MI_OK:
                self.keyIn = True
                self.keyValidCount = 2
                if self.key != uid:
                    self.key = uid
                    if uid is None:
                        return False
                    return True
        else:
            if self.keyIn:
                if self.keyValidCount > 0:
                    self.keyValidCount -= 1
                else:
                    self.keyIn = False
                    self.key = None
        return False

if __name__ == "__main__":
    reader = myRFIDReader(bus=0, dev=0)
    while True:
        if reader.Read():
            uid = uidToString(reader.key)
            print(uid, flush=True)
        time.sleep(0.01)
