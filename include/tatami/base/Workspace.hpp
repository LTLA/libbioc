#ifndef TATAMI_WORKSPACE_H
#define TATAMI_WORKSPACE_H

#include <memory>

/**
 * @file Workspace.hpp
 *
 * @brief Virtual class for workspaces.
 */

namespace tatami {

/**
 * @brief Virtual workspace class.
 *
 * Workspaces allow `Matrix` instances to persist information across certain method calls within the same thread,
 * allowing implementations to cache bits and pieces of information for non-random access patterns.
 *
 * Instances of this class cannot be constructed; it is only provided as a base class for `Matrix::new_workspace()` methods to return appropriate objects.
 */
class Workspace {
public:
    virtual ~Workspace() = default;

    // Defining the other constructors for rule of 5. The move constructors
    // don't get auto-made when a destructor is declared, and if I'm defining
    // them, I might as well define the copy constructors.  Technically I
    // suppose I don't need them because this class is just an interface, but
    // who knows if I'll add some move-able stuff here later.

    /**
     * Default move constructor.
     */
    Workspace(Workspace&&) = default;

    /**
     * Default move assignment operator.
     */
    Workspace& operator=(Workspace&&) = default;

    /**
     * Default copy constructor.
     */
    Workspace(const Workspace&) = default;

    /**
     * Default copy assignment operator.
     */
    Workspace& operator=(const Workspace&) = default;

protected:
    Workspace() = default;
};

}

#endif
