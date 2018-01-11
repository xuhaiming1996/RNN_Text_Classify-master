def assign_new_max_target_sen_num(self, session, max_target_sen_num_value):
    session.run(self._max_target_sen_num_update, feed_dict={self.new_max_target_sen_num: max_target_sen_num_value})


def assign_new_max_target_word_num(self, session, max_target_word_num_value):
    session.run(self._max_target_word_num_update, feed_dict={self.new_max_target_word_num: max_target_word_num_value})