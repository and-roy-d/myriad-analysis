
import sys, time
import qsghw.core.helpers as corehelpers
import qsghw.core.interfaces as interfaces
import qsghw.core.packetparser as packetparser
import qsghw.fpgaip.helpers as fpgaiphelpers

(regaccess, debug, verbose) = (False, False, True)



(ctrlurl, dataurl, freq, ampl, toread) = (sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]))



(ctrlif, dataif, eids) = corehelpers.open(ctrlurl, dataurl, opendata=True, quiet=True) #verbose=False)
print("cmd =", " ".join(sys.argv))

fset = fpgaiphelpers.makeinstance(ctrlif, eids, ":fullset:", verbose=debug)
tonegen = fpgaiphelpers.makeinstance(ctrlif, eids, ":tonegen:", verbose=debug)
frontend = fpgaiphelpers.makeinstance(ctrlif, eids, interfaces.channelizer, verbose=debug)
backend = fpgaiphelpers.makeinstance(ctrlif, eids, ":backend:", verbose=debug)
assert fset and tonegen and frontend and backend
fset.debug = tonegen.debug = frontend.debug = backend.debug = (regaccess, debug, verbose)



# route sync signal (the settings below are the default)
fset.write_sync5sel(5) # tonegen's sync-Q signal (syncsrc bit #5) to syncbus #5
fset.write_sync5mode(0) # pass through



# backend output selection: disable normal output, enable ATAN1 on monitor output
backend.write_pktrst(1)
backend.write_pktsrc("None", "pktdata")
backend.write_pktrst(0)

backend.write_monrst(1)
backend.write_pktsrc("ATAN1", "monitor")
backend.write_monmerge(32)
backend.write_mondec(1)
backend.write_monrst(0)



# set up module signal generation: sine wave on tone generator's 2nd ('Q') output
samplerate = tonegen.samplerate = frontend.samplerate
iqorder = tonegen.get_iqorder()
print("samplerate = %8.3f MHz   iqorder = %d   %s" % (samplerate/1e6, iqorder, tonegen))
tonegen.set_q(freq, ampl, phase=0., mode="ddstaylor-q", duty=0.25)
tonegen.dumpfields()



# collect data
time.sleep(0.1)
dataif.reset()
(nbytes, buffer) = dataif.readall(toread)
print(nbytes, buffer[0:31])

(headeroffsets, payloadoffsets, payloadlengths, seqnos) = packetparser.findlongestcontinuous(buffer, incr=1024)
(consumed, alldata) = packetparser.parsemany(buffer, headeroffsets[0], payloadoffsets, verbose = True)

print("longest continuous:", toread, "->", nbytes, "->", consumed, "=", len(payloadoffsets), "packets =", sum(payloadlengths), "byte total payload")
print(alldata.labels)
print(alldata.data.shape, "x", alldata.data.dtype)
print()

print(alldata.data)






#    (isum, qsum, psum) = (None, None, None)
#    for block in alldata.data:
#        i = block['i'].astype(int)
#        q = block['q'].astype(int)
#        p = i*i + q*q
#        isum = i if isum is None else isum + i
#        qsum = q if qsum is None else qsum + q
#        psum = p if psum is None else psum + p
##        for n in range(len(i)): print("~~~~~ %4d %10d %10d" % (n, i[n], q[n]))
##    print("~~~~~")
#
#    for i in range(len(psum)):
#        if psum[i] <= 0: continue
#        dB = 10. * math.log10(psum[i])
#        atan2 = math.atan2(isum, qsum)
#        #        print("+++++ %12.6f %12.6f %4d %4d   %5d   %20d %10.3f" % (freqlist[0]/1e6, realfreqlist[0]/1e6, i, binnolist[i], len(alldata.data), psum[i], dB))
#        print("+++++ %12.6f %12.6f %4d %4d   %5d   %20d %11.8f   %11.8f" % (freqlist[0]/1e6, realfreqlist[0]/1e6, i, (binnolist[i]+512)%1024, len(alldata.data), psum[i], dB, atan2))
#    print("+++++")
#
#    print()

