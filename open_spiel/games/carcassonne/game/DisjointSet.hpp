#include <algorithm>
#include <array>
#include <cstddef>

template <typename T, typename MergeFunc, std::size_t N>
class DisjointSet {
  private:
    mutable std::array<int, N> parent;
    std::array<int, N> rank;
    std::array<T, N> data;
    MergeFunc mergeLogic;

  public:
    explicit DisjointSet(MergeFunc mergeFunc) : mergeLogic(mergeFunc) {
        for (std::size_t i = 0; i < N; ++i) {
            parent[i] = static_cast<int>(i);
            rank[i] = 0;
            data[i] = T{};
        }
    }

    DisjointSet(const std::array<T, N> &initial_data, MergeFunc mergeFunc) : data(initial_data), mergeLogic(mergeFunc) {
        for (std::size_t i = 0; i < N; ++i) {
            parent[i] = static_cast<int>(i);
            rank[i] = 0;
        }
    }

    int find(int i) {
        if (parent[i] == i) {
            return i;
        }
        return parent[i] = find(parent[i]);
    }

    int find(int i) const {
        if (parent[i] == i) {
            return i;
        }
        return parent[i] = find(parent[i]);
    }

    T &getSetData(int i) { return data[find(i)]; }
    const T &getSetData(int i) const { return data[find(i)]; }

    void unionSet(int i, int j) {
        int root_i = find(i);
        int root_j = find(j);

        if (root_i != root_j) {
            T merged_data = mergeLogic(data[root_i], data[root_j]);

            if (rank[root_i] < rank[root_j]) {
                parent[root_i] = root_j;
                data[root_j] = merged_data;
            } else if (rank[root_i] > rank[root_j]) {
                parent[root_j] = root_i;
                data[root_i] = merged_data;
            } else {
                parent[root_j] = root_i;
                rank[root_i]++;
                data[root_i] = merged_data;
            }
        }
    }

    bool isConnected(int i, int j) { return find(i) == find(j); }
    bool isConnected(int i, int j) const { return find(i) == find(j); }

    class RootIterator {
      private:
        DisjointSet *ds;
        std::size_t index;

        void advanceToNextRoot() {
            while (index < N && ds->parent[index] != static_cast<int>(index)) {
                index++;
            }
        }

      public:
        RootIterator(DisjointSet *ds, std::size_t start_index) : ds(ds), index(start_index) { advanceToNextRoot(); }

        RootIterator &operator++() {
            index++;
            advanceToNextRoot();
            return *this;
        }

        RootIterator operator++(int) {
            RootIterator temp = *this;
            ++(*this);
            return temp;
        }

        T &operator*() { return ds->data[index]; }

        T *operator->() { return &(ds->data[index]); }

        int getIndex() const { return static_cast<int>(index); }

        bool operator==(const RootIterator &other) const { return index == other.index && ds == other.ds; }

        bool operator!=(const RootIterator &other) const { return !(*this == other); }
    };

    RootIterator begin() { return RootIterator(this, 0); }

    RootIterator end() { return RootIterator(this, N); }
};
