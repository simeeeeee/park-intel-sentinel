#!/usr/bin/env python3
# -*- coding: utf8 -*-
#
#    Copyright 2018 Daniel Perron
#
#    Base on Mario Gomez <mario.gomez@teubi.co>   MFRC522-Python
#
#    This file use part of MFRC522-Python
#    MFRC522-Python is a simple Python implementation for
#    the MFRC522 NFC Card Reader for the Raspberry Pi.
#
#    MFRC522-Python is free software:
#    you can redistribute it and/or modify
#    it under the terms of
#    the GNU Lesser General Public License as published by the
#    Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MFRC522-Python is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the
#    GNU Lesser General Public License along with MFRC522-Python.
#    If not, see <http://www.gnu.org/licenses/>.
#

from MFRC522 import MFRC522
import signal
import time

continue_reading = True


# function to read uid an conver it to a string

def uidToString(uid):
    mystring = ""
    for i in uid:
        mystring = format(i, '02X') + mystring
    return mystring


# Capture SIGINT for cleanup when the script is aborted
def end_read(signal, frame):
    global continue_reading
    print("Ctrl+C captured, ending read.")
    continue_reading = False

# Hook the SIGINT
signal.signal(signal.SIGINT, end_read)


# create a class contaings the RFID and only report when something is new

class myRFIDReader(MFRC522):
    def __init__(self,bus=0,dev=0):
        super().__init__(bus=bus,dev=dev)
        self.key = None
        self.keyIn = False
        self.keyValidCount=0;

    def Read(self):
        status, TagType = self.MFRC522_Request(super().PICC_REQIDL)
        if status == self.MI_OK:
            status, uid = self.MFRC522_SelectTagSN()
            if status == self.MI_OK:
                self.keyIn=True
                self.keyValidCount=2
                if self.key != uid:
                   self.key = uid
                   if uid is None:
                      return False
                   return True
        else:
            if self.keyIn:
                if self.keyValidCount>0:
                   self.keyValidCount= self.keyValidCount - 1
                else:
                   self.keyIn=False
                   self.key=None
        return False



reader1 = myRFIDReader(bus=0,dev=0)
reader2 = myRFIDReader(bus=0,dev=1)
reader3 = myRFIDReader(bus=10,dev=0)


# Welcome message
print("Welcome to the MFRC522 data read example")
print("Press Ctrl-C to stop.")

# This loop keeps checking for chips.
# If one is near it will get the UID and authenticate
while continue_reading:

    if reader1.Read():
       print("Reader1 : %s" %uidToString(reader1.key))
    if reader2.Read():
       print("Reader2 : %s" %uidToString(reader2.key))
    if reader3.Read():
       print("Reader3 : %s" %uidToString(reader3.key))

    time.sleep(0.010)