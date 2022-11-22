/*
Header-only DeePMD-kit C++ 11 library

This header-only library provides a C++ 11 interface to the DeePMD-kit C API.
*/

#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <cassert>

#include "c_api.h"

template <typename FPTYPE>
inline void _DP_DeepPotCompute(
    DP_DeepPot *dp,
    const int natom,
    const FPTYPE *coord,
    const int *atype,
    const FPTYPE *cell,
    double *energy,
    FPTYPE *force,
    FPTYPE *virial,
    FPTYPE *atomic_energy,
    FPTYPE *atomic_virial);

template <>
inline void _DP_DeepPotCompute<double>(
    DP_DeepPot *dp,
    const int natom,
    const double *coord,
    const int *atype,
    const double *cell,
    double *energy,
    double *force,
    double *virial,
    double *atomic_energy,
    double *atomic_virial)
{
    DP_DeepPotCompute(dp, natom, coord, atype, cell, energy, force, virial, atomic_energy, atomic_virial);
}

template <>
inline void _DP_DeepPotCompute<float>(
    DP_DeepPot *dp,
    const int natom,
    const float *coord,
    const int *atype,
    const float *cell,
    double *energy,
    float *force,
    float *virial,
    float *atomic_energy,
    float *atomic_virial)
{
    DP_DeepPotComputef(dp, natom, coord, atype, cell, energy, force, virial, atomic_energy, atomic_virial);
}

template <typename FPTYPE>
inline void _DP_DeepPotComputeNList(
    DP_DeepPot *dp,
    const int natom,
    const FPTYPE *coord,
    const int *atype,
    const FPTYPE *cell,
    const int nghost,
    const DP_Nlist *nlist,
    const int ago,
    double *energy,
    FPTYPE *force,
    FPTYPE *virial,
    FPTYPE *atomic_energy,
    FPTYPE *atomic_virial);

template <>
inline void _DP_DeepPotComputeNList<double>(
    DP_DeepPot *dp,
    const int natom,
    const double *coord,
    const int *atype,
    const double *cell,
    const int nghost,
    const DP_Nlist *nlist,
    const int ago,
    double *energy,
    double *force,
    double *virial,
    double *atomic_energy,
    double *atomic_virial)
{
    DP_DeepPotComputeNList(dp, natom, coord, atype, cell, nghost, nlist, ago, energy, force, virial, atomic_energy, atomic_virial);
}

template <>
inline void _DP_DeepPotComputeNList<float>(
    DP_DeepPot *dp,
    const int natom,
    const float *coord,
    const int *atype,
    const float *cell,
    const int nghost,
    const DP_Nlist *nlist,
    const int ago,
    double *energy,
    float *force,
    float *virial,
    float *atomic_energy,
    float *atomic_virial)
{
    DP_DeepPotComputeNListf(dp, natom, coord, atype, cell, nghost, nlist, ago, energy, force, virial, atomic_energy, atomic_virial);
}

template <typename FPTYPE>
inline void _DP_DeepPotModelDeviComputeNList(
    DP_DeepPotModelDevi *dp,
    const int natom,
    const FPTYPE *coord,
    const int *atype,
    const FPTYPE *cell,
    const int nghost,
    const DP_Nlist *nlist,
    const int ago,
    double *energy,
    FPTYPE *force,
    FPTYPE *virial,
    FPTYPE *atomic_energy,
    FPTYPE *atomic_virial);

template <>
inline void _DP_DeepPotModelDeviComputeNList<double>(
    DP_DeepPotModelDevi *dp,
    const int natom,
    const double *coord,
    const int *atype,
    const double *cell,
    const int nghost,
    const DP_Nlist *nlist,
    const int ago,
    double *energy,
    double *force,
    double *virial,
    double *atomic_energy,
    double *atomic_virial)
{
    DP_DeepPotModelDeviComputeNList(dp, natom, coord, atype, cell, nghost, nlist, ago, energy, force, virial, atomic_energy, atomic_virial);
}

template <>
inline void _DP_DeepPotModelDeviComputeNList<float>(
    DP_DeepPotModelDevi *dp,
    const int natom,
    const float *coord,
    const int *atype,
    const float *cell,
    const int nghost,
    const DP_Nlist *nlist,
    const int ago,
    double *energy,
    float *force,
    float *virial,
    float *atomic_energy,
    float *atomic_virial)
{
    DP_DeepPotModelDeviComputeNListf(dp, natom, coord, atype, cell, nghost, nlist, ago, energy, force, virial, atomic_energy, atomic_virial);
}

namespace deepmd
{
    namespace hpp
    {
        /**
        * @brief Neighbor list.
        **/
        struct InputNlist
        {
            InputNlist () 
                : inum(0), ilist(nullptr), numneigh(nullptr), firstneigh(nullptr),
                nl(DP_NewNlist(0, nullptr, nullptr, nullptr))
            {};
            InputNlist(
                int inum_,
                int *ilist_,
                int *numneigh_,
                int **firstneigh_)
                : inum(inum_), ilist(ilist_), numneigh(numneigh_), firstneigh(firstneigh_),
                nl(DP_NewNlist(inum_, ilist_, numneigh_, firstneigh_))
            {};
            /// @brief C API neighbor list.
            DP_Nlist* nl;
            /// @brief Number of core region atoms
            int inum;
            /// @brief Array stores the core region atom's index
            int *ilist;
            /// @brief Array stores the core region atom's neighbor atom number
            int *numneigh;
            /// @brief Array stores the core region atom's neighbor index
            int **firstneigh;
        };

        /**
         * @brief Convert pbtxt to pb.
         * @param[in] fn_pb_txt Filename of the pb txt file.
         * @param[in] fn_pb Filename of the pb file.
         **/
        void
        inline
        convert_pbtxt_to_pb(std::string fn_pb_txt, std::string fn_pb)
        {
            DP_ConvertPbtxtToPb(fn_pb_txt.c_str(), fn_pb.c_str());
        };
        /**
         * @brief Convert int vector to InputNlist.
         * @param[out] to_nlist InputNlist.
         * @param[in] from_nlist 2D int vector. The first axis represents the centeral atoms
         *                      and the second axis represents the neighbor atoms.
        */
        void
        inline
        convert_nlist(
            InputNlist & to_nlist,
            std::vector<std::vector<int> > & from_nlist
            )
        {
            to_nlist.inum = from_nlist.size();
            for(int ii = 0; ii < to_nlist.inum; ++ii){
                to_nlist.ilist[ii] = ii;
                to_nlist.numneigh[ii] = from_nlist[ii].size();
                to_nlist.firstneigh[ii] = &from_nlist[ii][0];
            }
            to_nlist.nl = DP_NewNlist(
                to_nlist.inum,
                to_nlist.ilist,
                to_nlist.numneigh,
                to_nlist.firstneigh
                );
        }
        /**
         * @brief Deep Potential.
         **/
        class DeepPot
        {
        public:
            /**
             * @brief DP constructor without initialization.
             **/
            DeepPot() : dp(nullptr) {};
            ~DeepPot(){};
            /**
             * @brief DP constructor with initialization.
             * @param[in] model The name of the frozen model file.
             **/
            DeepPot(const std::string &model) : dp(nullptr)
            {
                init(model);
            };
            /**
             * @brief Initialize the DP.
             * @param[in] model The name of the frozen model file.
             **/
            void init(const std::string &model)
            {
                if (dp)
                {
                    std::cerr << "WARNING: deepmd-kit should not be initialized twice, do nothing at the second call of initializer" << std::endl;
                    return;
                }
                dp = DP_NewDeepPot(model.c_str());
            };

            /**
             * @brief Evaluate the energy, force and virial by using this DP.
             * @param[out] ener The system energy.
             * @param[out] force The force on each atom.
             * @param[out] virial The virial.
             * @param[in] coord The coordinates of atoms. The array should be of size nframes x natoms x 3.
             * @param[in] atype The atom types. The list should contain natoms ints.
             * @param[in] box The cell of the region. The array should be of size nframes x 9 (PBC) or empty (no PBC).
             **/
            template <typename VALUETYPE>
            void compute(double &ener,
                         std::vector<VALUETYPE> &force,
                         std::vector<VALUETYPE> &virial,
                         const std::vector<VALUETYPE> &coord,
                         const std::vector<int> &atype,
                         const std::vector<VALUETYPE> &box)
            {
                unsigned int natoms = atype.size();
                assert(natoms * 3 == coord.size());
                if (!box.empty()) {
                    assert(box.size() == 9);
                }
                const VALUETYPE *coord_ = &coord[0];
                const VALUETYPE *box_ = !box.empty() ? &box[0] : nullptr;
                const int *atype_ = &atype[0];
                double *ener_ = &ener;
                force.resize(natoms * 3);
                virial.resize(9);
                VALUETYPE *force_ = &force[0];
                VALUETYPE *virial_ = &virial[0];

                _DP_DeepPotCompute<VALUETYPE>(dp, natoms, coord_, atype_, box_, ener_, force_, virial_, nullptr, nullptr);
            };
            /**
             * @brief Evaluate the energy, force, virial, atomic energy, and atomic virial by using this DP.
             * @param[out] ener The system energy.
             * @param[out] force The force on each atom.
             * @param[out] virial The virial.
             * @param[out] atom_energy The atomic energy.
             * @param[out] atom_virial The atomic virial.
             * @param[in] coord The coordinates of atoms. The array should be of size nframes x natoms x 3.
             * @param[in] atype The atom types. The list should contain natoms ints.
             * @param[in] box The cell of the region. The array should be of size nframes x 9 (PBC) or empty (no PBC).
             **/
            template <typename VALUETYPE>
            void compute(double &ener,
                         std::vector<VALUETYPE> &force,
                         std::vector<VALUETYPE> &virial,
                         std::vector<VALUETYPE> &atom_energy,
                         std::vector<VALUETYPE> &atom_virial,
                         const std::vector<VALUETYPE> &coord,
                         const std::vector<int> &atype,
                         const std::vector<VALUETYPE> &box)
            {
                unsigned int natoms = atype.size();
                assert(natoms * 3 == coord.size());
                if (!box.empty()) {
                    assert(box.size() == 9);
                }
                const VALUETYPE *coord_ = &coord[0];
                const VALUETYPE *box_ = !box.empty() ? &box[0] : nullptr;
                const int *atype_ = &atype[0];

                double *ener_ = &ener;
                force.resize(natoms * 3);
                virial.resize(9);
                atom_energy.resize(natoms);
                atom_virial.resize(natoms * 9);
                VALUETYPE *force_ = &force[0];
                VALUETYPE *virial_ = &virial[0];
                VALUETYPE *atomic_ener_ = &atom_energy[0];
                VALUETYPE *atomic_virial_ = &atom_virial[0];

                _DP_DeepPotCompute<VALUETYPE>(dp, natoms, coord_, atype_, box_, ener_, force_, virial_, atomic_ener_, atomic_virial_);
            };

            /**
             * @brief Evaluate the energy, force and virial by using this DP with the neighbor list.
             * @param[out] ener The system energy.
             * @param[out] force The force on each atom.
             * @param[out] virial The virial.
             * @param[in] coord The coordinates of atoms. The array should be of size nframes x natoms x 3.
             * @param[in] atype The atom types. The list should contain natoms ints.
             * @param[in] box The cell of the region. The array should be of size nframes x 9 (PBC) or empty (no PBC).
             * @param[in] nghost The number of ghost atoms.
             * @param[in] nlist The neighbor list.
             * @param[in] ago Update the internal neighbour list if ago is 0.
             **/
            template <typename VALUETYPE>
            void compute(double &ener,
                         std::vector<VALUETYPE> &force,
                         std::vector<VALUETYPE> &virial,
                         const std::vector<VALUETYPE> &coord,
                         const std::vector<int> &atype,
                         const std::vector<VALUETYPE> &box,
                         const int nghost,
                         const InputNlist &lmp_list,
                         const int &ago)
            {
                unsigned int natoms = atype.size();
                assert(natoms * 3 == coord.size());
                if (!box.empty())
                {
                    assert(box.size() == 9);
                }
                const VALUETYPE *coord_ = &coord[0];
                const VALUETYPE *box_ = !box.empty() ? &box[0] : nullptr;
                const int *atype_ = &atype[0];
                double *ener_ = &ener;
                force.resize(natoms * 3);
                virial.resize(9);
                VALUETYPE *force_ = &force[0];
                VALUETYPE *virial_ = &virial[0];

                _DP_DeepPotComputeNList<VALUETYPE>(dp, natoms, coord_, atype_, box_, nghost, lmp_list.nl, ago, ener_, force_, virial_, nullptr, nullptr);
            };
            /**
             * @brief Evaluate the energy, force, virial, atomic energy, and atomic virial by using this DP with the neighbor list.
             * @param[out] ener The system energy.
             * @param[out] force The force on each atom.
             * @param[out] virial The virial.
             * @param[out] atom_energy The atomic energy.
             * @param[out] atom_virial The atomic virial.
             * @param[in] coord The coordinates of atoms. The array should be of size nframes x natoms x 3.
             * @param[in] atype The atom types. The list should contain natoms ints.
             * @param[in] box The cell of the region. The array should be of size nframes x 9 (PBC) or empty (no PBC).
             * @param[in] nghost The number of ghost atoms.
             * @param[in] nlist The neighbor list.
             * @param[in] ago Update the internal neighbour list if ago is 0.
             **/
            template <typename VALUETYPE>
            void compute(double &ener,
                         std::vector<VALUETYPE> &force,
                         std::vector<VALUETYPE> &virial,
                         std::vector<VALUETYPE> &atom_energy,
                         std::vector<VALUETYPE> &atom_virial,
                         const std::vector<VALUETYPE> &coord,
                         const std::vector<int> &atype,
                         const std::vector<VALUETYPE> &box,
                         const int nghost,
                         const InputNlist &lmp_list,
                         const int &ago)
            {
                unsigned int natoms = atype.size();
                assert(natoms * 3 == coord.size());
                if (!box.empty())
                {
                    assert(box.size() == 9);
                }
                const VALUETYPE *coord_ = &coord[0];
                const VALUETYPE *box_ = !box.empty() ? &box[0] : nullptr;
                const int *atype_ = &atype[0];

                double *ener_ = &ener;
                force.resize(natoms * 3);
                virial.resize(9);
                atom_energy.resize(natoms);
                atom_virial.resize(natoms * 9);
                VALUETYPE *force_ = &force[0];
                VALUETYPE *virial_ = &virial[0];
                VALUETYPE *atomic_ener_ = &atom_energy[0];
                VALUETYPE *atomic_virial_ = &atom_virial[0];

                _DP_DeepPotComputeNList<VALUETYPE>(dp, natoms, coord_, atype_, box_, nghost, lmp_list.nl, ago, ener_, force_, virial_, atomic_ener_, atomic_virial_);
            };
            /**
             * @brief Get the cutoff radius.
             * @return The cutoff radius.
             **/
            double cutoff() const
            {
                assert(dp);
                return DP_DeepPotGetCutoff(dp);
            };
            /**
             * @brief Get the number of types.
             * @return The number of types.
             **/
            int numb_types() const
            {
                assert(dp);
                return DP_DeepPotGetNumbTypes(dp);
            };
            /**
             * @brief Get the type map (element name of the atom types) of this model.
             * @param[out] type_map The type map of this model.
             **/
            void get_type_map(std::string &type_map)
            {
                const char *type_map_c = DP_DeepPotGetTypeMap(dp);
                type_map.assign(type_map_c);
                delete[] type_map_c;
            };

        private:
            DP_DeepPot *dp;
        };

        /**
         * @brief Deep Potential model deviation.
         **/
        class DeepPotModelDevi
        {
        public:
            /**
             * @brief DP model deviation constructor without initialization.
             **/
            DeepPotModelDevi() : dp(nullptr) {};
            ~DeepPotModelDevi(){};
            /**
             * @brief DP model deviation constructor with initialization.
             * @param[in] models The names of the frozen model file.
             **/
            DeepPotModelDevi(const std::vector<std::string> &models) : dp(nullptr)
            {
                init(models);
            };
            /**
             * @brief Initialize the DP model deviation.
             * @param[in] model The name of the frozen model file.
             **/
            void init(const std::vector<std::string> &models)
            {
                if (dp)
                {
                    std::cerr << "WARNING: deepmd-kit should not be initialized twice, do nothing at the second call of initializer" << std::endl;
                    return;
                }
                std::vector<const char*> cstrings;
                cstrings.reserve(models.size());
                for (std::string const& str : models)
                    cstrings.push_back(str.data());

                dp = DP_NewDeepPotModelDevi(cstrings.data(), cstrings.size());
                numb_models = models.size();
            };

            /**
             * @brief Evaluate the energy, force and virial by using this DP model deviation.
             * @param[out] ener The system energy.
             * @param[out] force The force on each atom.
             * @param[out] virial The virial.
             * @param[in] coord The coordinates of atoms. The array should be of size nframes x natoms x 3.
             * @param[in] atype The atom types. The list should contain natoms ints.
             * @param[in] box The cell of the region. The array should be of size nframes x 9 (PBC) or empty (no PBC).
             **/
            template <typename VALUETYPE>
            void compute(std::vector<double> &ener,
                         std::vector<std::vector<VALUETYPE>> &force,
                         std::vector<std::vector<VALUETYPE>> &virial,
                         const std::vector<VALUETYPE> &coord,
                         const std::vector<int> &atype,
                         const std::vector<VALUETYPE> &box,
                         const int nghost,
                         const InputNlist &lmp_list,
                         const int &ago)
            {
                unsigned int natoms = atype.size();
                assert(natoms * 3 == coord.size());
                if (!box.empty()) {
                    assert(box.size() == 9);
                }
                const VALUETYPE *coord_ = &coord[0];
                const VALUETYPE *box_ = !box.empty() ? &box[0] : nullptr;
                const int *atype_ = &atype[0];

                // memory will be continous for std::vector but not std::vector<std::vector>
                std::vector<double> energy_flat(numb_models);
                std::vector<VALUETYPE> force_flat(numb_models * natoms * 3);
                std::vector<VALUETYPE> virial_flat(numb_models * 9);
                VALUETYPE *ener_ = &energy_flat[0];
                VALUETYPE *force_ = &force_flat[0];
                VALUETYPE *virial_ = &virial_flat[0];

                _DP_DeepPotModelDeviComputeNList<VALUETYPE>(dp, natoms, coord_, atype_, box_, nghost, lmp_list.nl, ago, ener_, force_, virial_, nullptr, nullptr);

                // reshape
                ener.resize(numb_models);
                force.resize(numb_models);
                virial.resize(numb_models);
                for (int i = 0; i < numb_models; i++)
                {
                    ener[i] = energy_flat[i];
                    force[i].resize(natoms * 3);
                    virial[i].resize(9);
                    for (int j = 0; j < natoms * 3; j++)
                        force[i][j] = force_flat[i * natoms * 3 + j];
                    for (int j = 0; j < 9; j++)
                        virial[i][j] = virial_flat[i * 9 + j];
                }
            };
            /**
             * @brief Evaluate the energy, force, virial, atomic energy, and atomic virial by using this DP model deviation.
             * @param[out] ener The system energy.
             * @param[out] force The force on each atom.
             * @param[out] virial The virial.
             * @param[out] atom_energy The atomic energy.
             * @param[out] atom_virial The atomic virial.
             * @param[in] coord The coordinates of atoms. The array should be of size nframes x natoms x 3.
             * @param[in] atype The atom types. The list should contain natoms ints.
             * @param[in] box The cell of the region. The array should be of size nframes x 9 (PBC) or empty (no PBC).
             **/
            template <typename VALUETYPE>
            void compute(std::vector<double> &ener,
                         std::vector<std::vector<VALUETYPE>> &force,
                         std::vector<std::vector<VALUETYPE>> &virial,
                         std::vector<std::vector<VALUETYPE>> &atom_energy,
                         std::vector<std::vector<VALUETYPE>> &atom_virial,
                         const std::vector<VALUETYPE> &coord,
                         const std::vector<int> &atype,
                         const std::vector<VALUETYPE> &box,
                         const int nghost,
                         const InputNlist &lmp_list,
                         const int &ago)
            {
                unsigned int natoms = atype.size();
                assert(natoms * 3 == coord.size());
                if (!box.empty()) {
                    assert(box.size() == 9);
                }
                const VALUETYPE *coord_ = &coord[0];
                const VALUETYPE *box_ = !box.empty() ? &box[0] : nullptr;
                const int *atype_ = &atype[0];

                std::vector<double> energy_flat(numb_models);
                std::vector<VALUETYPE> force_flat(numb_models * natoms * 3);
                std::vector<VALUETYPE> virial_flat(numb_models * 9);
                std::vector<VALUETYPE> atom_energy_flat(numb_models * natoms);
                std::vector<VALUETYPE> atom_virial_flat(numb_models * natoms * 9);
                VALUETYPE *ener_ = &energy_flat[0];
                VALUETYPE *force_ = &force_flat[0];
                VALUETYPE *virial_ = &virial_flat[0];
                VALUETYPE *atomic_ener_ = &atom_energy_flat[0];
                VALUETYPE *atomic_virial_ = &atom_virial_flat[0];

                _DP_DeepPotModelDeviComputeNList<VALUETYPE>(dp, natoms, coord_, atype_, box_, nghost, lmp_list.nl, ago, ener_, force_, virial_, atomic_ener_, atomic_virial_);

                // reshape
                ener.resize(numb_models);
                force.resize(numb_models);
                virial.resize(numb_models);
                atom_energy.resize(numb_models);
                atom_virial.resize(numb_models);
                for (int i = 0; i < numb_models; i++)
                {
                    ener[i] = energy_flat[i];
                    force[i].resize(natoms * 3);
                    virial[i].resize(9);
                    atom_energy[i].resize(natoms);
                    atom_virial[i].resize(natoms * 9);
                    for (int j = 0; j < natoms * 3; j++)
                        force[i][j] = force_flat[i * natoms * 3 + j];
                    for (int j = 0; j < 9; j++)
                        virial[i][j] = virial_flat[i * 9 + j];
                    for (int j = 0; j < natoms; j++)
                        atom_energy[i][j] = atom_energy_flat[i * natoms + j];
                    for (int j = 0; j < natoms * 9; j++)
                        atom_virial[i][j] = atom_virial_flat[i * natoms * 9 + j];
                }
            };
            /**
             * @brief Get the cutoff radius.
             * @return The cutoff radius.
             **/
            double cutoff() const
            {
                assert(dp);
                return DP_DeepPotModelDeviGetCutoff(dp);
            };
            /**
             * @brief Get the number of types.
             * @return The number of types.
             **/
            int numb_types() const
            {
                assert(dp);
                return DP_DeepPotModelDeviGetNumbTypes(dp);
            };

        private:
            DP_DeepPotModelDevi *dp;
            int numb_models;
        };
    }
}