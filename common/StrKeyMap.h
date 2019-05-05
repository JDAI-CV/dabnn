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
    using map_t = std::map<std::string, V>;
    map_t map_;

   public:
    inline V &operator[](const std::string &key) {
        return map_[key];
    }
    inline const V &at(const std::string &key) const {
        try {
            return map_.at(key);
        } catch (const std::out_of_range &e) {
            throw std::out_of_range("Key " + key + " not found.");
        }
    }
    // This is "const decltype(as_const(map_).begin()) begin() const"
    const typename map_t::const_iterator begin() const {
        return map_.begin();
    }
    const typename map_t::const_iterator end() const {
        return map_.end();
    }
    void clear() {
        map_.clear();
    }
    const typename map_t::const_iterator find(const std::string &key) const {
        return map_.find(key);
    }
    size_t size() const {
        return map_.size();
    }
    bool has(const std::string &key) const {
        return map_.find(key) != map_.end();
    }

    void insert(const std::pair<std::string, V> &p) {
        map_.insert(p);
    }
};

#endif  // DNNLIBRARY_DNN_MAP_H
