import numpy as np

class ContactData():

    ''' Contains all info in one keystroke
        
    Attributes:
        id: a unique integer to number a keystroke with upper limit 16
        state: 1 => the start of a keystroke, 2 => middle, 3 => the end
            number of states that equal 1 or 3 can only be 1
        label: -1 => unsure, 0 => negative, 1 => positive
    '''

    def __init__(self, id = 0, state = 0, x = 0, y = 0, area = 0, force = 0, major = 0, minor = 0,
            delta_x = 0, delta_y = 0, delta_force = 0, delta_area = 0, label = 0, frame_id = 0):
        self.id: int = id
        self.state: int = state
        self.x: float = x
        self.y: float = y
        self.area: float = area
        self.force: float = force
        self.major: float = major
        self.minor: float = minor
        self.delta_x: float = delta_x
        self.delta_y: float = delta_y
        self.delta_force: float = delta_force
        self.delta_area: float = delta_area
        self.label: int = label
        self.frame_id: int = frame_id

    def __str__(self):
        contact_info = 'id: ' + str(self.id) + '\n' + 'state: ' + str(self.state) + '\n' \
            + 'x: ' + str(self.x) + '\n' + 'y: ' + str(self.y) + '\n' \
            +'area: ' + str(self.area) + '\n' + 'force: ' + str(self.force) + '\n' \
            + 'major: ' + str(self.major) + '\n' + 'minor: ' + str(self.minor) + '\n' \
            + 'delta_x: ' + str(self.delta_x) + '\n' + 'delta_y: ' + str(self.delta_y) + '\n' \
            + 'delta_force: ' + str(self.delta_force) + '\n' + 'delta_area: ' + str(self.delta_area) + '\n' \
            + 'label: ' + str(self.label) + '\n' + 'frame_id: ' + str(self.frame_id)
        return contact_info
    
    def to_numpy(self):
        return np.array([self.id, self.state, self.x, self.y, self.area, self.force, self.major, self.minor,
            self.delta_x, self.delta_y, self.delta_force, self.delta_area, self.label, self.frame_id], dtype=np.float64)
    
    def type_str(self):
        type_info = 'id: ' + str(type(self.id)) + '\n' + 'state: ' + str(type(self.state)) + '\n' \
            + 'x: ' + str(type(self.x)) + '\n' + 'y: ' + str(type(self.y)) + '\n' \
            +'area: ' + str(type(self.area)) + '\n' + 'force: ' + str(type(self.force)) + '\n' \
            + 'major: ' + str(type(self.major)) + '\n' + 'minor: ' + str(type(self.minor)) + '\n' \
            + 'delta_x: ' + str(type(self.delta_x)) + '\n' + 'delta_y: ' + str(type(self.delta_y)) + '\n' \
            + 'delta_force: ' + str(type(self.delta_force)) + '\n' + 'delta_area: ' + str(type(self.delta_area)) + '\n' \
            + 'label: ' + str(type(self.label)) + '\n' + 'frame_id: ' + str(type(self.frame_id))
        return type_info


class FrameData():
    ''' info in one frame of morph board
    attrs:
        force_array: TODO
        contacts: list of keystroke contacts on the board
        timestamp: time acquired from time.perf_counter for each frame
    '''

    # MAGNI = 4
    # layout = utils.Layout()

    def __init__(self, force_array, timestamp):
        self.force_array = force_array
        self.timestamp = timestamp
        self.contacts = []
    
    def append_contact(self, contact):
        self.contacts.append(contact)

    def to_numpy(self):
        ''' convert frame data to numpy array
        return:
            return a numpy array of force_array
        '''
        return self.force_array

    # def output(self) -> np.array:
    #     ''' add contacts to force_map and show image
    #     return:
    #         return force_map image with marked contacts
    #     '''
    #     MAGNI = FrameData.MAGNI
    #     layout = FrameData.layout
    #     H, W = self.force_array.shape
    #     force_map = cv2.resize(self.force_array, (W * MAGNI, H * MAGNI)).astype(float)
    #     force_map = cv2.cvtColor(force_map, cv2.COLOR_GRAY2BGR)

    #     color = (150, 150, 150)
    #     for key, rec in layout.key_pos.items():
    #         pt1, pt2 = rec
    #         cv2.rectangle(force_map, (int(pt1[0] * W * MAGNI), int(pt1[1] * H * MAGNI)),
    #             (int(pt2[0] * W * MAGNI), int(pt2[1] * H * MAGNI)), color, 1)
    #         # use len(key) // 2 to modify key offset
    #         cv2.putText(force_map, key, org=(int((pt1[0] + 0.03 - (len(key) // 2) * 0.01) * W * MAGNI),
    #             int((pt1[1] + 0.05) * H * MAGNI)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #             fontScale=0.15*MAGNI, color=color, thickness=1)

    #     for contact in self.contacts:
    #         color = (0, 0, 0)
    #         if contact.label == -1:     # Unsure
    #             color = (255, 0, 0)
    #         if contact.label == 1:      # Positive sample
    #             color = (0, 255, 0)
    #         if contact.label == 0:      # Negative sample
    #             color = (0, 0, 255)
    #         cv2.circle(force_map, (int((contact.x * W + 0.5) * MAGNI), int((contact.y * H + 0.5) * MAGNI)), 8, color, -1)
    #     return force_map