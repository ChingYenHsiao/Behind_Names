import pickle
from name_module.Chinese_name_separate import *


class FortuneMapCalculater:
    main_dict = {}
    addition_dict = {}

    def __init__(self, main_dict=None, addition_dict=None):
        self.stroke_state_dict = {1: '吉', 2: '凶', 3: '吉', 4: '凶', 5: '吉', 6: '吉', 7: '吉', 8: '吉', 9: '凶', 10: '凶',
                                  11: '吉', 12: '凶', 13: '吉', 14: '凶', 15: '吉', 16: '吉', 17: '吉', 18: '吉', 19: '凶',
                                  20: '凶',
                                  21: '吉', 22: '凶', 23: '吉', 24: '吉', 25: '吉', 26: '凶帶吉', 27: '吉帶凶', 28: '凶', 29: '吉',
                                  30: '吉帶凶',
                                  31: '吉', 32: '吉', 33: '吉', 34: '凶', 35: '吉', 36: '凶', 37: '吉', 38: '凶帶吉', 39: '吉',
                                  40: '吉帶凶',
                                  41: '吉', 42: '吉帶凶', 43: '吉帶凶', 44: '凶', 45: '吉', 46: '凶', 47: '吉', 48: '吉', 49: '凶',
                                  50: '吉帶凶',
                                  51: '吉帶凶', 52: '吉', 53: '吉帶凶', 54: '凶', 55: '吉帶凶', 56: '凶', 57: '凶帶吉', 58: '吉帶凶',
                                  59: '凶',
                                  60: '凶',
                                  61: '吉帶凶', 62: '凶', 63: '吉', 64: '凶', 65: '吉', 66: '凶', 67: '吉', 68: '吉', 69: '凶',
                                  70: '凶',
                                  71: '吉帶凶', 72: '凶', 73: '吉', 74: '凶', 75: '吉帶凶', 76: '凶', 77: '吉帶凶', 78: '吉帶凶',
                                  79: '凶',
                                  80: '吉帶凶', 81: '吉'
                                  }
        self.talent_states = {'木木木': '大吉', '木木火': '大吉', '木木土': '大吉', '木木金': '凶多吉少', '木木水': '吉多於凶', '木火木': '大吉',
                              '木火火': '中吉',
                              '木火土': '大吉', '木火金': '凶多於吉', '木火水': '大凶', '木土木': '大凶', '木土火': '中吉', '木土土': '吉',
                              '木土金': '吉多於凶',
                              '木土水': '大凶', '木金木': '大凶', '木金火': '大凶', '木金土': '凶多於吉', '木金金': '大凶', '木金水': '大凶',
                              '木水木': '大吉',
                              '木水火': '凶多於吉', '木水土': '凶多於吉', '木水金': '大吉', '木水水': '大吉', '火木木': '大吉', '火木火': '大吉',
                              '火木土': '大吉',
                              '火木金': '凶多於吉', '火木水': '中吉', '火火木': '大吉', '火火火': '中吉', '火火土': '大吉', '火火金': '大凶',
                              '火火水': '大凶',
                              '火土木': '吉多於凶', '火土火': '大吉', '火土土': '大吉', '火土金': '大吉', '火土水': '吉多於凶', '火金木': '大凶',
                              '火金火': '大凶',
                              '火金土': '吉凶參半', '火金金': '大凶', '火金水': '大凶', '火水木': '凶多於吉', '火水火': '大凶', '火水土': '大凶',
                              '火水金': '大凶',
                              '火水水': '大凶', '土木木': '中吉', '土木火': '中吉', '土木土': '凶多於吉', '土木金': '大凶', '土木水': '凶多於吉',
                              '土火木': '大吉',
                              '土火火': '大吉', '土火土': '大吉', '土火金': '吉多於凶', '土火水': '大凶', '土土木': '中吉', '土土火': '大吉',
                              '土土土': '大吉',
                              '土土金': '大吉', '土土水': '凶多於吉', '土金木': '凶多於吉', '土金火': '凶多於吉', '土金土': '大吉', '土金金': '大吉',
                              '土金水': '大吉',
                              '土水木': '凶多於吉', '土水火': '大凶', '土水土': '大凶', '土水金': '吉凶參半', '土水水': '大凶', '金木木': '凶多於吉',
                              '金木火': '凶多於吉',
                              '金木土': '凶多於吉', '金木金': '大凶', '金木水': '凶多於吉', '金火木': '凶多於吉', '金火火': '吉凶參半', '金火土': '吉凶參半',
                              '金火金': '大凶',
                              '金火水': '大凶', '金土木': '中吉', '金土火': '大吉', '金土土': '大吉', '金土金': '大吉', '金土水': '吉多於凶',
                              '金金木': '大凶', '金金土': '大吉', '金金金': '中吉', '金金水': '中吉', '金水木': '大吉', '金水火': '凶多於吉',
                              '金水土': '吉',
                              '金水金': '大吉', '金水水': '中吉', '水木木': '大吉', '水木火': '大吉', '水木土': '大吉', '水木金': '凶多於吉',
                              '水木水': '大吉',
                              '水火木': '中吉', '水火火': '大凶', '水火土': '凶多於吉', '水火金': '大凶', '水火水': '大凶', '水土木': '大凶',
                              '水土火': '中吉',
                              '水土土': '中吉', '水土金': '中吉', '水土水': '大凶', '水金木': '凶多於吉', '水金火': '凶多於吉', '水金土': '大吉',
                              '水金金': '中吉',
                              '水金水': '大吉', '水水木': '大吉', '水水火': '大凶', '水水土': '大凶', '水水金': '大吉', '金金火': '吉', '水水水': '中吉'}
        # 三才 天才 人才 地才
        self.talent_types = {1: '木', 2: '木', 3: '火', 4: '火',
                             5: '土', 6: '土', 7: '金', 8: '金', 9: '水', 0: '水'}

        if main_dict is not None:
            self.main_dict = main_dict
        else:
            self.main_dict = {}
        if addition_dict is not None:
            self.addition_dict = addition_dict
        else:
            self.addition_dict = {}

    def load_main_dict(self, path):
        """ load main dict from file

        Args:
            path (path): path to main dict
        """
        if path.exists():
            with open(path, 'rb') as handle:
                self.main_dict = pickle.loads(handle.read())

    def load_addition_dict(self, path):
        """ load addition dict rom file

        Args:
            path (path): path to addition dict
        """
        if path.exists():
            with open(path, 'rb') as handle:
                self.addition_dict = pickle.loads(handle.read())

    def get_stroke(self, character):
        """  Get the stroke of a character.

        Args:
            character (str): 字

        Returns:
            stroke(int): stroke of the character
        """
        stroke = 0
        if character in self.main_dict:
            radical = self.main_dict[character]['radical']
            if radical == character:
                stroke = int((self.main_dict[character]['stroke_count']))
            else:
                stroke = int(
                    self.main_dict[character]['non_radical_stroke_count']) + self.get_stroke(radical)
        elif character in self.addition_dict:
            stroke = int(self.addition_dict[character]['stroke_count'])

        if stroke == 0:
            print('Error: ', character, "get stroke count failed!")
        return stroke

    def get_stroke_state(self, count):
        stroke_state = "不明"
        if count in self.stroke_state_dict:
            stroke_state = self.stroke_state_dict[count]
        return stroke_state

    def stroke_heaven(self, last_name):
        """ Get heaven(天格) stroke count.
            單姓	姓氏筆劃加1 	姓「王」	5
            複姓	姓氏筆劃相加總	姓「司馬」	15
        Args:
            last_name (str): Last Name

        Returns:
            heaven_stroke(int): heaven stroke count
        """
        heaven_stroke = 0
        if len(last_name) == 1:
            heaven_stroke = self.get_stroke(last_name) + 1
        elif len(last_name) == 2:
            heaven_stroke = self.get_stroke(
                last_name[0]) + self.get_stroke(last_name[1])
        return heaven_stroke

    def get_state_heaven(self, last_name):
        """ Get heaven(天格) stroke state.
        Args:
            last_name (str): Last Name
        Returns:
            heaven_stroke(int): heaven stroke state
        """
        heaven_stroke = self.stroke_heaven(last_name)
        return self.stroke_state_dict.get(heaven_stroke, "凶")

    def stroke_man(self, last_name, first_name):
        """ Get man(人格) stroke count.
            單姓	姓氏加名字第一字的筆劃	文  天祥 8
            複姓	姓氏最後一字加名字第一字的筆劃	司馬  光	16
        Args:
            last_name (str): Last Name
            first_name (str): First Name

        Returns:
            man_stroke(int): man stroke count
        """
        man_stroke = 0
        fn = first_name.strip()
        if len(last_name) == 1:
            man_stroke = self.get_stroke(last_name) + self.get_stroke(fn[0])
        elif len(last_name) == 2:
            man_stroke = self.get_stroke(last_name[1]) + self.get_stroke(fn[0])
        return man_stroke

    def get_state_man(self, last_name, first_name):
        """ Get man(人格) stroke state.
        Args:
            last_name (str): Last Name
            first_name (str): First name
        Returns:
            man_stroke(int): man stroke state
        """
        man_stroke = self.stroke_man(last_name, first_name)
        return self.stroke_state_dict.get(man_stroke, "凶")

    def stroke_earth(self, first_name):
        """ Get earth(地格) stroke count.
            單名	名字筆劃加1	王  二  3
            複名	名字的筆劃相加總	文  天祥	15
        Args:
            first_name (str): first name

        Returns:
            earth_stroke(int): earth stroke count
        """
        earth_stroke = 0
        fn = first_name.strip()
        if len(fn) == 1:
            earth_stroke = self.get_stroke(fn) + 1
        elif len(fn) >= 2:
            for character in fn:
                earth_stroke += self.get_stroke(character)
        return earth_stroke

    def get_state_earth(self, first_name):
        """ Get earth(地格) stroke state.
        Args:
            first_name (str): First Name
        Returns:
            earth_stroke(int): earth stroke state
        """
        earth_stroke = self.stroke_earth(first_name)
        return self.stroke_state_dict.get(earth_stroke, "凶")

    def stroke_outside(self, last_name, first_name):
        """ Get outside(外格) stroke count.
            單姓單名	等於2	項羽，岳飛 = 2
            單姓複名	名字最後一字加1	陶淵明 = 9
            複姓單名	姓氏第一字加1	司馬光 = 6
            複名複名	姓氏第一字加名字最後一字 = 司馬相如
        Args:
            last_name (str): Last Name
            first_name (str): First Name

        Returns:
            outside_stroke(int): outside stroke count
        """

        outside_stroke = 0
        fn = first_name.strip()
        if len(last_name) == 1 and len(fn) == 1:
            outside_stroke = 2
        elif len(last_name) == 1 and len(fn) == 2:
            outside_stroke = self.get_stroke(fn[-1]) + 1
        elif len(last_name) == 2 and len(fn) == 1:
            outside_stroke = self.get_stroke(last_name[0]) + 1
        elif len(last_name) == 2 and len(fn) == 2:
            outside_stroke = self.get_stroke(
                last_name[0]) + self.get_stroke(fn[1])
        return outside_stroke

    def get_state_outside(self, last_name, first_name):
        """ Get outside(外格) stroke state.
        Args:
            last_name (str): Last Name
            first_name (str): First Name
        Returns:
            outside_stroke(int): outside stroke state
        """
        outside_stroke = self.stroke_outside(last_name, first_name)
        return self.stroke_state_dict.get(outside_stroke, "凶")

    def stroke_total(self, last_name, first_name):
        """ Get total(總格) stroke count.
            八十一數之輪動，乃是以八十數為一單位，81起還本歸元 = 1, 如83劃-80=3劃
            姓與名字的筆劃全部相加總，如文天祥的總格19。
        Args:
            last_name (str): Last Name
            first_name (str): First Name

        Returns:
            total_stroke(int): total stroke count
        """
        total_stroke = 0

        fn = first_name.strip()
        for character in last_name:
            total_stroke += self.get_stroke(character)
        for character in fn:
            total_stroke += self.get_stroke(character)

        if total_stroke > 80:
            total_stroke -= 80
        return total_stroke

    def get_state_total(self, last_name, first_name):
        """ Get total(總格) stroke state.
        Args:
            last_name (str): Last Name
            first_name (str): First Name
        Returns:
            total_stroke(int): total stroke state
        """
        total_stroke = self.stroke_total(last_name, first_name)
        return self.stroke_state_dict.get(total_stroke, "凶")

    def get_talent_type(self, structure):
        # 三才 天才 人才 地才
        if isinstance(structure, int):
            return self.talent_types[structure % 10]
        else:
            return '水'

    def get_state_talent(self, last_name, first_name):
        element1 = self.get_talent_type(self.stroke_heaven(last_name))
        element2 = self.get_talent_type(self.stroke_earth(last_name))
        element3 = self.get_talent_type(self.stroke_man(last_name, first_name))
        talent_style = ''.join((element1, element2, element3))
        return self.talent_states.get(talent_style, "凶")

    def name_fortune_telling(self, name):
        last_name = get_last_name(name)
        first_name = get_first_name(name)
        print('名字：', name)
        print('五格')

        heaven = self.stroke_heaven(last_name)
        earth = self.stroke_earth(last_name)
        man = self.stroke_man(last_name, first_name)
        outside = self.stroke_outside(last_name, first_name)
        total = self.stroke_total(last_name, first_name)

        print('天格：', heaven, self.get_state_heaven(last_name))
        print('地格：', earth, self.get_state_earth(first_name))
        print('人格：', man, self.get_state_man(last_name, first_name))
        print('外格：', outside, self.get_state_outside(last_name, first_name))
        print('總格：', total, self.get_state_total(last_name, first_name))

        print('三才')
        print('天才:', self.get_talent_type(heaven))
        print('人才:', self.get_talent_type(man))
        print('地才:', self.get_talent_type(earth))
        print('三才格局：', self.get_state_talent(last_name, first_name))
        print('')
