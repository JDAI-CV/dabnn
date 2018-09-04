// Copyright 2019 JD.com Inc. JD AI

//
// Created by daquexian on 8/20/18.
//

#ifndef DNNLIBRARY_DNN_MAP_H
#define DNNLIBRARY_DNN_MAP_H

#include <map>
#include <string>

/**
 * std::map whose key is std::string, so that we can print the key when a
 * std::out_of_range is thrown
 * @tparam V the value typename
 */
template <typename V>
class StrKeyMap {
   private:
    std::map<std::string, V> map_;

   public:
    inline V &operator[](const std::string &key) { return map_[key]; }
    inline const V &at(const std::string &key) const {
        try {
            return map_.at(key);
        } catch (const std::out_of_range &e) {
            throw std::out_of_range("Key " + key + " not found.");
        }
    }
    auto begin() const { return map_.begin(); }
    auto end() const { return map_.end(); }
    auto clear() { return map_.clear(); }
    auto find(const std::string &key) { return map_.find(key); }
    auto size() const { return map_.size(); }
};

#endif  // DNNLIBRARY_DNN_MAP_H
