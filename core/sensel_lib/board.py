import time
import numpy as np
import multiprocessing as mp
import compress_pickle
try:
    from core.sensel_lib import sensel
    from core.sensel_lib.frame_data import FrameData
    from core.sensel_lib.frame_data import ContactData
except:
    import sensel
    from frame_data import FrameData
    from frame_data import ContactData
import _thread

def compress_dump_to_file(data, filename):
    with open(filename, 'wb') as f:
        compress_pickle.dump(data, f)

def save_frame(q_frames, q_insts, save_frame_filename):
    MAX_FRAME_SIZE = 1000
    segment = 0
    frames = []
    save_frame_filename = save_frame_filename.split(".gz")[0]
    current_dump_process = None
    try:
        while True:
            if not q_frames.empty():
                frame = q_frames.get()
                frames.append(frame)
                if len(frames) >= MAX_FRAME_SIZE:
                    to_save_frames = frames
                    current_dump_process = mp.Process(target=compress_dump_to_file, args=(to_save_frames, f"{save_frame_filename}_{segment}.gz")).start()
                    frames = []
                    segment += 1
            if not q_insts.empty():
                inst = q_insts.get()
                if inst == "flush":
                    # save_frame_file.flush()
                    pass
                if inst == "stop":
                    to_save_frames = frames
                    current_dump_process = mp.Process(target=compress_dump_to_file, args=(to_save_frames, f"{save_frame_filename}_{segment}.gz")).start()
                    segment = 0
                    frames = []
                if inst == "close":
                    # save_frame_file.close()
                    if current_dump_process:
                        current_dump_process.join()
                    break
                if inst == "change_filename":
                    frames = []
                    segment = 0
                    new_filename = q_insts.get()
                    save_frame_filename = new_filename.split(".gz")[0]
            time.sleep(0.001)
    except:
        if current_dump_process:
            current_dump_process.join()

class Board():
    MAX_X = 230.0
    MAX_Y = 130.0
    FPS = 50

    def __init__(self, save_filename=None, if_run=True, print_fps=True):
        self.save_frame_filename = f"{save_filename}" if save_filename else None
        self.is_recording = False
        self.if_run = if_run
        self.print_fps = print_fps
        self.frame_id = 0
        self.fps_cnt = 0
        self.fps_start_time = time.time()
        self._openSensel()
        self._initFrame()
        self.q_frames = mp.Queue() if self.save_frame_filename else None
        self.q_insts = mp.Queue() if self.save_frame_filename else None
        if self.save_frame_filename:
            self.save_frame_p = mp.Process(target=save_frame, args=(self.q_frames, self.q_insts, self.save_frame_filename))
            self.save_frame_p.start()

    def stop(self):
        self.flush()
        self.q_insts.put("close")
        if self.save_frame_filename:
            self.save_frame_p.join()
        self.is_running = False
    
    def start_record(self):
        self.is_recording = True

    def stop_record(self):
        self.q_insts.put("stop")
        self.is_recording = False

    def change_save_filename(self, save_filename):
        self.save_frame_filename = f"{save_filename}.gz"
        self.q_insts.put("change_filename")
        self.q_insts.put(self.save_frame_filename)

    def _openSensel(self):
        handle = None
        (error, device_list) = sensel.getDeviceList()
        if device_list.num_devices != 0:
            (error, handle) = sensel.openDeviceByID(device_list.devices[0].idx)
        self.handle = handle

    def _initFrame(self):
        (error, self.info) = sensel.getSensorInfo(self.handle)
        error = sensel.setFrameContent(self.handle, 0x0F)
        error = sensel.setContactsMask(self.handle, 0x0F)
        (error, frame) = sensel.allocateFrameData(self.handle)
        error = sensel.startScanning(self.handle)
        self._frame = frame
        self.frames = []
        self.updated = False
        if self.if_run:
            try:
                _thread.start_new_thread(self._run, ())
            except:
                print("Thread Error")

    def _closeSensel(self):
        self.is_running = False
        error = sensel.freeFrameData(self.handle, self._frame)
        error = sensel.stopScanning(self.handle)
        error = sensel.close(self.handle)
    
    def _sync(self):
        if len(self.frames) > 0:
            while ((int(time.time() * 1e6) - self.frames[-1].timestamp) * Board.FPS < 1):
                pass

    def _run(self):
        self.is_running = True
        while (self.is_running):
            error = sensel.readSensor(self.handle)
            (error, num_frames) = sensel.getNumAvailableFrames(self.handle)
            for i in range(num_frames):
                self._sync()
                timestamp = int(time.time() * 1e6)
                error = sensel.getFrame(self.handle, self._frame)
            R = self.info.num_rows
            C = self.info.num_cols
            force_array = np.zeros((R, C))
            for r in range(R):
                force_array[r, :] = self._frame.force_array[r * C : (r + 1) * C]
            force_array *= 0.2
            frame = FrameData(force_array, timestamp)

            for i in range(self._frame.n_contacts):
                c = self._frame.contacts[i]
                x = c.x_pos / Board.MAX_X
                y = c.y_pos / Board.MAX_Y
                contact = ContactData(c.id, c.state, x, y, c.area, c.total_force, c.major_axis, c.minor_axis, c.delta_x, c.delta_y, c.delta_force, c.delta_area, 0, self.frame_id)
                frame.append_contact(contact)

            if self.save_frame_filename and self.is_recording:
                self.q_frames.put(frame)
                self.flush()
            
            if self.print_fps:
                self._printFPS()
            self.frame_id += 1
            self.fps_cnt += 1
            self.updated = True
        
        self._closeSensel()

    def _printFPS(self, time_interval=1):
        if time.time() - self.fps_start_time > time_interval:
            print(f"Sensel FPS: {self.fps_cnt / (time.time() - self.fps_start_time)}")
            self.fps_start_time = time.time()
            self.fps_cnt = 0

    def flush(self):
        self.q_insts.put("flush")

    def getFrame(self):
        while (len(self.frames) == 0):
            time.sleep(0.001)
        return self.frames.pop(0)
    
    def getNewFrame(self):
        while self.updated == False:
            time.sleep(0.001)
        self.updated = False
        return self.frames[-1]
    
    def getFrameTime(self):
        if len(self.frames) >= 2:
            return round(self.frames[-1].timestamp - self.frames[-2].timestamp, 5)
        return 0
    
    def setScanDetail(self, detail):
        error = sensel.setScanDetail(self.handle, detail)
    
    def setMaxFrameRate(self, rate):
        error = sensel.setMaxFrameRate(self.handle, rate)
    
    def getScanDetail(self):
        return sensel.getScanDetail(self.handle)

    def getMaxFrameRate(self):
        return sensel.getMaxFrameRate(self.handle)


if __name__ == "__main__":
    board = Board(if_run=True, save_filename="test")
    
    # chagne board settings
    
    # board.setScanDetail(1)
    # board.setMaxFrameRate(500)
    # print(board.getScanDetail())
    # print(board.getMaxFrameRate())

    start_time = time.time()
    cnt = 0
    while time.time() - start_time < 10:
        time.sleep(0.1)
        print(time.time() - start_time)
    print("Done")
    board.stop()
    print("Stopped")