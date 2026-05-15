/*
 * -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*-
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Authors: This scenario was extended by an undergraduate research student from
 *          UFPA (Federal University of Pará) as part of Kleber Vilhena's Master's project.
 *          Base implementation by Andrea Lacava, Michele Polese, Matteo Bordin.
 */

/**
 * @file scenario-hierarchical-xangai.cc
 * @brief This scenario integrates Traffic Steering (TS) and Energy Saving (ES) mechanisms,
 * enhanced with Shanghai mobility modeling, and is controlled by a hierarchical RL agent.
 *
 * The code was developed by a scientific initiation student from UFPA (Federal University of Pará),
 * working under Kleber Vilhena's Master's project. It builds upon the original documentation
 * by adding specific implementations for handover and energy-saving scenarios with Shanghai mobility.
 *
 * Key features:
 * 1. Accepts control actions for forced handovers (TS) and cell ON/OFF state (ES) through a single
 *    control file, parsed by the ns-3 device according to the action header.
 * 2. Generates all required KPIs for TS (per-UE SINR, throughput) and ES (aggregated metrics).
 * 3. Implements the BsStateTrace function, which logs the ON/OFF state of cells to bsState.txt,
 *    essential for the ES agent's state observation.
 * 4. Incorporates Shanghai mobility patterns for realistic user movement simulation.
 * 5. Merges all configurable parameters from both original TS and ES scenarios.
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/applications-module.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/lte-enb-net-device.h"
#include "ns3/lte-enb-phy.h"
#include <ns3/lte-ue-net-device.h>
#include "ns3/mmwave-helper.h"
#include "ns3/mmwave-enb-net-device.h"
#include "ns3/mmwave-enb-phy.h"
#include "ns3/epc-helper.h"
#include "ns3/mmwave-point-to-point-epc-helper.h"
#include "ns3/lte-helper.h"
#include "ns3/energy-heuristic.h"
#include "ns3/mobility-module.h"
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <random>
#include <cmath>
#include <limits>

using namespace ns3;
using namespace mmwave;

NS_LOG_COMPONENT_DEFINE ("ScenarioHierarchical");

// ============================================================================
// Shanghai Mobility Implementation
// ============================================================================

/**
 * \brief Simple waypoint structure for Shanghai mobility
 */
struct SimpleWaypoint
{
    double time;
    double x, y, z;

    SimpleWaypoint (double t, double px, double py, double pz = 1.5)
        : time (t), x (px), y (py), z (pz) {}
};

/**
 * \brief Waypoint structure for UAV mobility with flight parameters
 */
struct UAVWaypoint
{
    double time;
    double x, y, z;
    double speed;

    UAVWaypoint (double t, double px, double py, double pz, double spd = 15.0)
        : time (t), x (px), y (py), z (pz), speed (spd) {}
};

/**
 * \brief UAV flight patterns for gNB mobility
 */
enum UAVFlightPattern
{
    HOVERING = 0,        // Hover at fixed position with GPS drift
    CIRCULAR_ORBIT = 1,  // Orbit around area of interest
    PATROL_LINEAR = 2,   // Back and forth between points
    FIGURE_EIGHT = 3,    // Figure-8 trajectory
    GRID_COVERAGE = 4,   // Lawnmower grid sweep
    ALTITUDE_VAR = 5     // Vertical oscillation pattern
};

/**
 * \brief Generate Shanghai-like urban mobility patterns
 * Adapted for the hierarchical scenario with configurable center position
 */
class ShanghaiMobilityGenerator
{
public:
    /**
     * \brief Generate trajectories for all UEs
     * \param ueCount Number of UEs
     * \param centerX Center X position of the scenario
     * \param centerY Center Y position of the scenario
     * \param areaRadius Radius of the area where UEs move
     * \param simTime Simulation time in seconds
     * \return Vector of trajectories (vector of waypoints for each UE)
     */
    static std::vector<std::vector<SimpleWaypoint>> GenerateTrajectories (
        uint32_t ueCount,
        double centerX,
        double centerY,
        double areaRadius,
        double simTime)
    {
        std::vector<std::vector<SimpleWaypoint>> trajectories;

        for (uint32_t ueId = 0; ueId < ueCount; ++ueId)
        {
            trajectories.push_back (GenerateUrbanTrajectory (ueId, centerX, centerY, areaRadius, simTime));
        }

        return trajectories;
    }

private:
    /**
     * \brief Generate urban trajectory for a single UE
     */
    static std::vector<SimpleWaypoint> GenerateUrbanTrajectory (
        uint32_t ueId,
        double centerX,
        double centerY,
        double areaRadius,
        double simTime)
    {
        std::vector<SimpleWaypoint> waypoints;

        // Calculate grid position for initial placement
        uint32_t gridSize = static_cast<uint32_t> (std::ceil (std::sqrt (static_cast<double> (ueId + 1))));
        if (gridSize < 3) gridSize = 3;

        // Distribute UEs in a grid pattern around the center
        double spacing = (2.0 * areaRadius) / gridSize;
        double baseX = centerX - areaRadius + (ueId % gridSize) * spacing + spacing / 2.0;
        double baseY = centerY - areaRadius + ((ueId / gridSize) % gridSize) * spacing + spacing / 2.0;

        // Ensure base position is within bounds
        baseX = std::max (centerX - areaRadius + 50, std::min (baseX, centerX + areaRadius - 50));
        baseY = std::max (centerY - areaRadius + 50, std::min (baseY, centerY + areaRadius - 50));

        // Different movement patterns based on UE ID
        uint32_t pattern = ueId % 6;

        // Calculate number of waypoints based on simulation time
        int numWaypoints = static_cast<int> (simTime / 0.4) + 1;
        if (numWaypoints < 10) numWaypoints = 10;
        if (numWaypoints > 500) numWaypoints = 500;

        // =================================================================
        // Shanghai VUR Dataset-based mobility patterns
        // Based on real trajectory data analysis:
        //   - Average speed: 15 km/h (4.2 m/s)
        //   - Median speed: 15.8 km/h (4.4 m/s)
        //   - Max speed: 43 km/h (12 m/s)
        //   - 75% of vehicles below 20 km/h
        // =================================================================
        switch (pattern)
        {
        case 0: // Urban main road (based on Shanghai dataset P75)
            {
                // Speed: 5.5 m/s (~20 km/h) - Shanghai P75 percentile
                double baseSpeed = 5.5;
                double currentX = baseX;
                double currentY = baseY;
                double currentTime = 0.0;

                while (currentTime < simTime)
                {
                    // Variable speed (15-25 km/h range)
                    double speed = baseSpeed + (std::sin(currentTime * 0.5 + ueId) * 1.5);

                    // Move for 3-5 seconds
                    int moveSteps = 8 + (ueId % 5);
                    for (int i = 0; i < moveSteps && currentTime < simTime; ++i)
                    {
                        currentTime += 0.4;
                        currentX += speed * 0.4;
                        if (currentX > centerX + areaRadius)
                            currentX = centerX - areaRadius + 50;
                        waypoints.emplace_back (currentTime, currentX, currentY);
                    }

                    // Stop at traffic light (2-4 seconds)
                    int stopSteps = 5 + (ueId % 5);
                    for (int i = 0; i < stopSteps && currentTime < simTime; ++i)
                    {
                        currentTime += 0.4;
                        waypoints.emplace_back (currentTime, currentX, currentY);
                    }
                }
                break;
            }
        case 1: // Urban slow with turns (Shanghai average ~15 km/h)
            {
                double speed = 4.2; // m/s (~15 km/h) - Shanghai average
                double currentX = baseX;
                double currentY = baseY;
                double currentTime = 0.0;
                int direction = 0; // 0=right, 1=up, 2=left, 3=down

                while (currentTime < simTime)
                {
                    // Move in current direction for ~100m
                    int steps = 20 + (ueId % 10);
                    for (int i = 0; i < steps && currentTime < simTime; ++i)
                    {
                        currentTime += 0.5;
                        switch (direction)
                        {
                            case 0: currentX += speed * 0.5; break;
                            case 1: currentY += speed * 0.5; break;
                            case 2: currentX -= speed * 0.5; break;
                            case 3: currentY -= speed * 0.5; break;
                        }
                        // Clamp to area
                        currentX = std::max(centerX - areaRadius + 20, std::min(currentX, centerX + areaRadius - 20));
                        currentY = std::max(centerY - areaRadius + 20, std::min(currentY, centerY + areaRadius - 20));
                        waypoints.emplace_back (currentTime, currentX, currentY);
                    }

                    // Stop at intersection (3-6 seconds)
                    int stopSteps = 6 + (ueId % 6);
                    for (int i = 0; i < stopSteps && currentTime < simTime; ++i)
                    {
                        currentTime += 0.5;
                        waypoints.emplace_back (currentTime, currentX, currentY);
                    }

                    // Turn (change direction)
                    direction = (direction + 1 + (ueId % 2)) % 4;
                }
                break;
            }
        case 2: // Intersection crossing - slow approach (Shanghai intersection data)
            {
                // Very slow at intersections: 3 m/s (~11 km/h)
                double speed = 3.0;
                double currentX = baseX;
                double currentY = baseY;
                double currentTime = 0.0;

                while (currentTime < simTime)
                {
                    // Approach intersection slowly
                    for (int i = 0; i < 15 && currentTime < simTime; ++i)
                    {
                        currentTime += 0.4;
                        currentY += speed * 0.4;
                        if (currentY > centerY + areaRadius)
                            currentY = centerY - areaRadius + 50;
                        waypoints.emplace_back (currentTime, currentX, currentY);
                    }

                    // Wait at intersection (5-10 seconds)
                    int waitSteps = 12 + (ueId % 12);
                    for (int i = 0; i < waitSteps && currentTime < simTime; ++i)
                    {
                        currentTime += 0.4;
                        waypoints.emplace_back (currentTime, currentX, currentY);
                    }

                    // Cross intersection faster
                    for (int i = 0; i < 8 && currentTime < simTime; ++i)
                    {
                        currentTime += 0.4;
                        currentY += 5.0 * 0.4; // Faster crossing
                        waypoints.emplace_back (currentTime, currentX, currentY);
                    }
                }
                break;
            }
        case 3: // Roundabout/curve movement (slow circular)
            {
                // Slow roundabout: 3.5 m/s (~12.6 km/h)
                double radius = 30.0 + (ueId % 4) * 15.0;
                double speed = 3.5;
                double angularSpeed = speed / radius;
                double currentTime = 0.0;
                double angle = (ueId % 8) * M_PI / 4; // Different starting angles

                while (currentTime < simTime)
                {
                    // Enter roundabout slowly
                    for (int i = 0; i < 5 && currentTime < simTime; ++i)
                    {
                        currentTime += 0.5;
                        waypoints.emplace_back(currentTime, baseX, baseY);
                    }

                    // Circle in roundabout
                    for (int i = 0; i < 40 && currentTime < simTime; ++i)
                    {
                        currentTime += 0.5;
                        angle += angularSpeed * 0.5;
                        double x = baseX + radius * std::cos(angle);
                        double y = baseY + radius * std::sin(angle);
                        x = std::max(centerX - areaRadius + 20, std::min(x, centerX + areaRadius - 20));
                        y = std::max(centerY - areaRadius + 20, std::min(y, centerY + areaRadius - 20));
                        waypoints.emplace_back(currentTime, x, y);
                    }
                }
                break;
            }
        case 4: // Stop-and-go dense traffic (most common in Shanghai - 47% of data)
            {
                // Very slow: 2.5 m/s (~9 km/h) with frequent stops
                double currentX = baseX;
                double currentY = baseY;
                double currentTime = 0.0;

                while (currentTime < simTime)
                {
                    // Short move (2-3 seconds at ~9 km/h)
                    int moveSteps = 4 + (ueId % 4);
                    for (int i = 0; i < moveSteps && currentTime < simTime; ++i)
                    {
                        currentTime += 0.5;
                        currentX += 2.5 * 0.5;
                        currentY += (ueId % 2 == 0 ? 0.5 : -0.3); // Slight Y drift
                        if (currentX > centerX + areaRadius) currentX = centerX - areaRadius + 50;
                        currentY = std::max(centerY - areaRadius + 20, std::min(currentY, centerY + areaRadius - 20));
                        waypoints.emplace_back(currentTime, currentX, currentY);
                    }

                    // Frequent stops (3-6 seconds)
                    int stopSteps = 6 + (ueId % 6);
                    for (int i = 0; i < stopSteps && currentTime < simTime; ++i)
                    {
                        currentTime += 0.5;
                        waypoints.emplace_back(currentTime, currentX, currentY);
                    }
                }
                break;
            }
        case 5: // Diagonal slow movement (typical urban ~15 km/h)
            {
                double speed = 4.0; // m/s (~14.4 km/h)
                double currentX = baseX;
                double currentY = baseY;
                double currentTime = 0.0;
                double angleRad = (0.5 + (ueId % 4) * 0.25) * M_PI / 4; // 22.5-67.5 degrees

                while (currentTime < simTime)
                {
                    // Move diagonally
                    for (int i = 0; i < 15 && currentTime < simTime; ++i)
                    {
                        currentTime += 0.4;
                        currentX += speed * 0.4 * std::cos(angleRad);
                        currentY += speed * 0.4 * std::sin(angleRad);
                        // Wrap around
                        if (currentX > centerX + areaRadius)
                            currentX = centerX - areaRadius + std::fmod(currentX - centerX - areaRadius, 2.0 * areaRadius);
                        if (currentY > centerY + areaRadius)
                            currentY = centerY - areaRadius + std::fmod(currentY - centerY - areaRadius, 2.0 * areaRadius);
                        if (currentX < centerX - areaRadius)
                            currentX = centerX + areaRadius - 50;
                        if (currentY < centerY - areaRadius)
                            currentY = centerY + areaRadius - 50;
                        waypoints.emplace_back(currentTime, currentX, currentY);
                    }

                    // Stop periodically
                    int stopSteps = 4 + (ueId % 5);
                    for (int i = 0; i < stopSteps && currentTime < simTime; ++i)
                    {
                        currentTime += 0.4;
                        waypoints.emplace_back(currentTime, currentX, currentY);
                    }

                    // Change direction slightly
                    angleRad += (ueId % 2 == 0 ? 0.3 : -0.3);
                }
                break;
            }
        }

        // Ensure at least one waypoint at time 0
        if (waypoints.empty () || waypoints[0].time > 0.0)
        {
            waypoints.insert (waypoints.begin (), SimpleWaypoint (0.0, baseX, baseY));
        }

        return waypoints;
    }
};

/**
 * \brief Setup Shanghai mobility for UE nodes using WaypointMobilityModel
 * \param ueNodes Container of UE nodes
 * \param centerX Center X position
 * \param centerY Center Y position
 * \param areaRadius Movement area radius
 * \param simTime Simulation time
 */
void
SetupShanghaiMobility (NodeContainer ueNodes, double centerX, double centerY, double areaRadius, double simTime)
{
    NS_LOG_FUNCTION ("Setting up Shanghai mobility for " << ueNodes.GetN () << " UEs");
    NS_LOG_INFO ("Shanghai mobility: center=(" << centerX << "," << centerY << "), radius=" << areaRadius << ", simTime=" << simTime);

    // Generate trajectories
    auto trajectories = ShanghaiMobilityGenerator::GenerateTrajectories (
        ueNodes.GetN (), centerX, centerY, areaRadius, simTime);

    // Setup WaypointMobilityModel for each UE
    for (uint32_t i = 0; i < ueNodes.GetN (); ++i)
    {
        Ptr<Node> ueNode = ueNodes.Get (i);
        Ptr<WaypointMobilityModel> mobility = CreateObject<WaypointMobilityModel> ();

        // Add waypoints
        const auto& waypoints = trajectories[i];
        NS_LOG_DEBUG ("UE " << i << " has " << waypoints.size () << " waypoints, pattern " << (i % 6));

        for (const auto& wp : waypoints)
        {
            Vector position (wp.x, wp.y, wp.z);
            Time waypointTime = Seconds (wp.time);
            Waypoint waypoint (waypointTime, position);
            mobility->AddWaypoint (waypoint);
        }

        ueNode->AggregateObject (mobility);
    }

    NS_LOG_INFO ("Shanghai mobility setup completed for " << ueNodes.GetN () << " UEs");
}

// ============================================================================
// End of Shanghai Mobility Implementation
// ============================================================================

// ============================================================================
// UAV Mobility Implementation for gNBs
// ============================================================================

/**
 * \brief Generate UAV trajectories for mmWave gNBs
 * Transforms ground-based gNBs into UAV-BS (drone base stations)
 */
class UAVTrajectoryGenerator
{
public:
    /**
     * \brief Generate trajectories for all UAV gNBs
     * \param gnbCount Number of gNBs (excluding LTE eNB)
     * \param basePositions Initial positions of gNBs (x, y from hexagonal layout)
     * \param centerX Center X position of the scenario
     * \param centerY Center Y position of the scenario
     * \param baseAltitude Base altitude for UAVs
     * \param altitudeVariation Maximum altitude variation
     * \param simTime Simulation time in seconds
     * \param pattern Flight pattern to use
     * \param maxSpeed Maximum UAV speed (m/s)
     * \param orbitRadius Radius for circular orbit pattern
     * \param patrolLength Length for patrol pattern
     * \param gridSize Size for grid coverage pattern
     * \return Vector of trajectories (vector of UAVWaypoints for each gNB)
     */
    static std::vector<std::vector<UAVWaypoint>> GenerateTrajectories (
        uint32_t gnbCount,
        const std::vector<Vector>& basePositions,
        double centerX,
        double centerY,
        double baseAltitude,
        double altitudeVariation,
        double simTime,
        UAVFlightPattern pattern,
        double maxSpeed,
        double orbitRadius,
        double patrolLength,
        double gridSize)
    {
        std::vector<std::vector<UAVWaypoint>> trajectories;

        for (uint32_t gnbId = 0; gnbId < gnbCount; ++gnbId)
        {
            Vector basePos = basePositions[gnbId];

            switch (pattern)
            {
                case HOVERING:
                    trajectories.push_back (GenerateHoveringPattern (
                        gnbId, basePos.x, basePos.y, baseAltitude, simTime));
                    break;
                case CIRCULAR_ORBIT:
                    trajectories.push_back (GenerateCircularPattern (
                        gnbId, basePos.x, basePos.y, baseAltitude, simTime, maxSpeed, orbitRadius));
                    break;
                case PATROL_LINEAR:
                    trajectories.push_back (GeneratePatrolPattern (
                        gnbId, basePos.x, basePos.y, baseAltitude, simTime, maxSpeed, patrolLength));
                    break;
                case FIGURE_EIGHT:
                    trajectories.push_back (GenerateFigure8Pattern (
                        gnbId, basePos.x, basePos.y, baseAltitude, simTime, maxSpeed, orbitRadius));
                    break;
                case GRID_COVERAGE:
                    trajectories.push_back (GenerateGridPattern (
                        gnbId, basePos.x, basePos.y, baseAltitude, simTime, maxSpeed, gridSize));
                    break;
                case ALTITUDE_VAR:
                    trajectories.push_back (GenerateAltitudeVariationPattern (
                        gnbId, basePos.x, basePos.y, baseAltitude, altitudeVariation, simTime, maxSpeed, orbitRadius));
                    break;
                default:
                    trajectories.push_back (GenerateHoveringPattern (
                        gnbId, basePos.x, basePos.y, baseAltitude, simTime));
                    break;
            }
        }

        return trajectories;
    }

private:
    /**
     * \brief Generate hovering pattern with GPS drift (±1-2m)
     */
    static std::vector<UAVWaypoint> GenerateHoveringPattern (
        uint32_t gnbId,
        double baseX,
        double baseY,
        double baseAltitude,
        double simTime)
    {
        std::vector<UAVWaypoint> waypoints;

        // Seed based on gnbId for reproducibility
        std::mt19937 rng (gnbId + 12345);
        std::uniform_real_distribution<double> drift (-1.5, 1.5); // GPS drift ±1.5m

        double waypointInterval = 1.0; // Update position every 1 second
        int numWaypoints = static_cast<int> (simTime / waypointInterval) + 1;

        for (int i = 0; i < numWaypoints; ++i)
        {
            double time = i * waypointInterval;
            if (time > simTime) break;

            double x = baseX + drift (rng);
            double y = baseY + drift (rng);
            double z = baseAltitude + drift (rng) * 0.5; // Smaller vertical drift

            waypoints.emplace_back (time, x, y, z, 1.0); // Low speed for hovering
        }

        // Ensure waypoint at time 0
        if (waypoints.empty () || waypoints[0].time > 0.0)
        {
            waypoints.insert (waypoints.begin (), UAVWaypoint (0.0, baseX, baseY, baseAltitude, 1.0));
        }

        return waypoints;
    }

    /**
     * \brief Generate circular orbit pattern around base position
     */
    static std::vector<UAVWaypoint> GenerateCircularPattern (
        uint32_t gnbId,
        double baseX,
        double baseY,
        double baseAltitude,
        double simTime,
        double maxSpeed,
        double orbitRadius)
    {
        std::vector<UAVWaypoint> waypoints;

        // Calculate angular speed based on radius and linear speed
        double circumference = 2.0 * M_PI * orbitRadius;
        double orbitPeriod = circumference / maxSpeed;
        double angularSpeed = 2.0 * M_PI / orbitPeriod;

        // Phase offset based on gnbId to avoid collisions
        double phaseOffset = (gnbId * 2.0 * M_PI) / 7.0;

        double waypointInterval = 0.5; // Update every 0.5 seconds for smooth orbit
        int numWaypoints = static_cast<int> (simTime / waypointInterval) + 1;

        for (int i = 0; i < numWaypoints; ++i)
        {
            double time = i * waypointInterval;
            if (time > simTime) break;

            double angle = angularSpeed * time + phaseOffset;
            double x = baseX + orbitRadius * std::cos (angle);
            double y = baseY + orbitRadius * std::sin (angle);
            double z = baseAltitude;

            waypoints.emplace_back (time, x, y, z, maxSpeed);
        }

        if (waypoints.empty () || waypoints[0].time > 0.0)
        {
            double x = baseX + orbitRadius * std::cos (phaseOffset);
            double y = baseY + orbitRadius * std::sin (phaseOffset);
            waypoints.insert (waypoints.begin (), UAVWaypoint (0.0, x, y, baseAltitude, maxSpeed));
        }

        return waypoints;
    }

    /**
     * \brief Generate linear patrol pattern (north-south or east-west)
     */
    static std::vector<UAVWaypoint> GeneratePatrolPattern (
        uint32_t gnbId,
        double baseX,
        double baseY,
        double baseAltitude,
        double simTime,
        double maxSpeed,
        double patrolLength)
    {
        std::vector<UAVWaypoint> waypoints;

        // Alternate patrol direction based on gnbId
        bool eastWest = (gnbId % 2 == 0);

        double halfLength = patrolLength / 2.0;
        double legTime = patrolLength / maxSpeed;
        double waypointInterval = 0.5;

        double currentTime = 0.0;

        while (currentTime <= simTime)
        {
            double progress = std::fmod (currentTime, 2.0 * legTime);
            double displacement;

            if (progress < legTime)
            {
                // Moving in positive direction
                displacement = -halfLength + (progress / legTime) * patrolLength;
            }
            else
            {
                // Moving in negative direction
                displacement = halfLength - ((progress - legTime) / legTime) * patrolLength;
            }

            double x = eastWest ? baseX + displacement : baseX;
            double y = eastWest ? baseY : baseY + displacement;

            waypoints.emplace_back (currentTime, x, y, baseAltitude, maxSpeed);
            currentTime += waypointInterval;
        }

        if (waypoints.empty () || waypoints[0].time > 0.0)
        {
            double x = eastWest ? baseX - halfLength : baseX;
            double y = eastWest ? baseY : baseY - halfLength;
            waypoints.insert (waypoints.begin (), UAVWaypoint (0.0, x, y, baseAltitude, maxSpeed));
        }

        return waypoints;
    }

    /**
     * \brief Generate figure-8 pattern with two connected lobes
     */
    static std::vector<UAVWaypoint> GenerateFigure8Pattern (
        uint32_t gnbId,
        double baseX,
        double baseY,
        double baseAltitude,
        double simTime,
        double maxSpeed,
        double lobeRadius)
    {
        std::vector<UAVWaypoint> waypoints;

        // Figure-8 is parametric: x = sin(t), y = sin(2t)/2
        // Scale by lobeRadius
        double patternPeriod = (4.0 * M_PI * lobeRadius) / maxSpeed; // Approximate path length
        double angularSpeed = 2.0 * M_PI / patternPeriod;

        double phaseOffset = (gnbId * 2.0 * M_PI) / 7.0;
        double waypointInterval = 0.5;

        double currentTime = 0.0;
        while (currentTime <= simTime)
        {
            double t = angularSpeed * currentTime + phaseOffset;
            double x = baseX + lobeRadius * std::sin (t);
            double y = baseY + lobeRadius * std::sin (2.0 * t) / 2.0;

            waypoints.emplace_back (currentTime, x, y, baseAltitude, maxSpeed);
            currentTime += waypointInterval;
        }

        if (waypoints.empty () || waypoints[0].time > 0.0)
        {
            double t = phaseOffset;
            double x = baseX + lobeRadius * std::sin (t);
            double y = baseY + lobeRadius * std::sin (2.0 * t) / 2.0;
            waypoints.insert (waypoints.begin (), UAVWaypoint (0.0, x, y, baseAltitude, maxSpeed));
        }

        return waypoints;
    }

    /**
     * \brief Generate lawnmower grid coverage pattern
     */
    static std::vector<UAVWaypoint> GenerateGridPattern (
        uint32_t gnbId,
        double baseX,
        double baseY,
        double baseAltitude,
        double simTime,
        double maxSpeed,
        double gridSize)
    {
        std::vector<UAVWaypoint> waypoints;

        double halfGrid = gridSize / 2.0;
        double stripWidth = gridSize / 5.0; // 5 strips per coverage
        double waypointInterval = 0.5;

        // Starting corner based on gnbId
        double startX = baseX - halfGrid;
        double startY = baseY - halfGrid;

        double currentTime = 0.0;
        double currentX = startX;
        double currentY = startY;
        int stripIndex = 0;
        bool movingUp = true;

        while (currentTime <= simTime)
        {
            waypoints.emplace_back (currentTime, currentX, currentY, baseAltitude, maxSpeed);

            // Calculate next position
            if (movingUp)
            {
                currentY += maxSpeed * waypointInterval;
                if (currentY >= baseY + halfGrid)
                {
                    currentY = baseY + halfGrid;
                    movingUp = false;
                    stripIndex++;
                    currentX = startX + stripIndex * stripWidth;
                    if (currentX > baseX + halfGrid)
                    {
                        currentX = startX;
                        stripIndex = 0;
                    }
                }
            }
            else
            {
                currentY -= maxSpeed * waypointInterval;
                if (currentY <= baseY - halfGrid)
                {
                    currentY = baseY - halfGrid;
                    movingUp = true;
                    stripIndex++;
                    currentX = startX + stripIndex * stripWidth;
                    if (currentX > baseX + halfGrid)
                    {
                        currentX = startX;
                        stripIndex = 0;
                    }
                }
            }

            currentTime += waypointInterval;
        }

        if (waypoints.empty () || waypoints[0].time > 0.0)
        {
            waypoints.insert (waypoints.begin (), UAVWaypoint (0.0, startX, startY, baseAltitude, maxSpeed));
        }

        return waypoints;
    }

    /**
     * \brief Generate altitude variation pattern with sinusoidal vertical oscillation
     */
    static std::vector<UAVWaypoint> GenerateAltitudeVariationPattern (
        uint32_t gnbId,
        double baseX,
        double baseY,
        double baseAltitude,
        double altitudeVariation,
        double simTime,
        double maxSpeed,
        double orbitRadius)
    {
        std::vector<UAVWaypoint> waypoints;

        // Combine horizontal circular motion with vertical sinusoidal oscillation
        double circumference = 2.0 * M_PI * orbitRadius;
        double orbitPeriod = circumference / maxSpeed;
        double angularSpeed = 2.0 * M_PI / orbitPeriod;

        // Vertical oscillation period (different from horizontal)
        double verticalPeriod = orbitPeriod * 0.5; // Faster vertical oscillation
        double verticalAngularSpeed = 2.0 * M_PI / verticalPeriod;

        double phaseOffset = (gnbId * 2.0 * M_PI) / 7.0;
        double waypointInterval = 0.5;

        double currentTime = 0.0;
        while (currentTime <= simTime)
        {
            double angle = angularSpeed * currentTime + phaseOffset;
            double x = baseX + orbitRadius * std::cos (angle);
            double y = baseY + orbitRadius * std::sin (angle);

            // Sinusoidal altitude variation
            double z = baseAltitude + altitudeVariation * std::sin (verticalAngularSpeed * currentTime);

            waypoints.emplace_back (currentTime, x, y, z, maxSpeed);
            currentTime += waypointInterval;
        }

        if (waypoints.empty () || waypoints[0].time > 0.0)
        {
            double x = baseX + orbitRadius * std::cos (phaseOffset);
            double y = baseY + orbitRadius * std::sin (phaseOffset);
            waypoints.insert (waypoints.begin (), UAVWaypoint (0.0, x, y, baseAltitude, maxSpeed));
        }

        return waypoints;
    }
};

/**
 * \brief Setup UAV mobility for mmWave gNB nodes using WaypointMobilityModel
 * \param mmWaveEnbNodes Container of mmWave gNB nodes
 * \param basePositions Initial base positions (x, y) for each gNB
 * \param centerX Center X position of scenario
 * \param centerY Center Y position of scenario
 * \param baseAltitude Base altitude for UAVs (meters)
 * \param altitudeVariation Maximum altitude variation (meters)
 * \param simTime Simulation time in seconds
 * \param pattern Flight pattern to use
 * \param maxSpeed Maximum UAV speed (m/s)
 * \param orbitRadius Radius for circular/figure-8 patterns
 * \param patrolLength Length for patrol pattern
 * \param gridSize Size for grid coverage pattern
 */
void
SetupUAVMobility (NodeContainer mmWaveEnbNodes,
                  const std::vector<Vector>& basePositions,
                  double centerX,
                  double centerY,
                  double baseAltitude,
                  double altitudeVariation,
                  double simTime,
                  UAVFlightPattern pattern,
                  double maxSpeed,
                  double orbitRadius,
                  double patrolLength,
                  double gridSize)
{
    NS_LOG_FUNCTION ("Setting up UAV mobility for " << mmWaveEnbNodes.GetN () << " gNBs");
    NS_LOG_INFO ("UAV mobility: altitude=" << baseAltitude << "m, pattern=" << pattern
                 << ", speed=" << maxSpeed << "m/s");

    // Generate trajectories for all gNBs
    auto trajectories = UAVTrajectoryGenerator::GenerateTrajectories (
        mmWaveEnbNodes.GetN (),
        basePositions,
        centerX,
        centerY,
        baseAltitude,
        altitudeVariation,
        simTime,
        pattern,
        maxSpeed,
        orbitRadius,
        patrolLength,
        gridSize);

    // Setup WaypointMobilityModel for each gNB
    for (uint32_t i = 0; i < mmWaveEnbNodes.GetN (); ++i)
    {
        Ptr<Node> gnbNode = mmWaveEnbNodes.Get (i);
        Ptr<WaypointMobilityModel> mobility = CreateObject<WaypointMobilityModel> ();

        const auto& waypoints = trajectories[i];
        NS_LOG_DEBUG ("gNB " << i << " has " << waypoints.size () << " waypoints");

        for (const auto& wp : waypoints)
        {
            Vector position (wp.x, wp.y, wp.z);
            Time waypointTime = Seconds (wp.time);
            Waypoint waypoint (waypointTime, position);
            mobility->AddWaypoint (waypoint);
        }

        gnbNode->AggregateObject (mobility);
    }

    NS_LOG_INFO ("UAV mobility setup completed for " << mmWaveEnbNodes.GetN () << " gNBs");
}

// ============================================================================
// End of UAV Mobility Implementation
// ============================================================================

// --- Trace de Mobilidade ---
std::ofstream mobilityTraceFile;

/**
 * \brief Callback para registrar mudanças de posição dos UEs
 */
void
MobilityTraceCallback (std::ostream *os, std::string context, Ptr<const MobilityModel> mobility)
{
    Vector pos = mobility->GetPosition ();

    // Extrai o Node ID do contexto (formato: "/NodeList/X/$ns3::MobilityModel/CourseChange")
    size_t firstSlash = context.find ("/", 1);
    size_t secondSlash = context.find ("/", firstSlash + 1);
    std::string nodeIdStr = context.substr (firstSlash + 1, secondSlash - firstSlash - 1);

    *os << Simulator::Now ().GetSeconds () << "\t"
        << nodeIdStr << "\t"
        << pos.x << "\t" << pos.y << "\t" << pos.z << std::endl;
}

/**
 * \brief Função para registrar posições de todos os UEs periodicamente
 * Garante dados contínuos para animação independente do modelo de mobilidade
 */
void
PeriodicMobilityTrace (std::ostream *os, NodeContainer ueNodes, double interval, double endTime)
{
    double currentTime = Simulator::Now ().GetSeconds ();

    for (uint32_t i = 0; i < ueNodes.GetN (); ++i)
    {
        Ptr<Node> node = ueNodes.Get (i);
        Ptr<MobilityModel> mobility = node->GetObject<MobilityModel> ();
        if (mobility)
        {
            Vector pos = mobility->GetPosition ();
            *os << currentTime << "\t"
                << "UE\t"
                << node->GetId () << "\t"
                << pos.x << "\t" << pos.y << "\t" << pos.z << std::endl;
        }
    }

    // Agenda próxima execução se ainda houver tempo
    if (currentTime + interval < endTime)
    {
        Simulator::Schedule (Seconds (interval), &PeriodicMobilityTrace, os, ueNodes, interval, endTime);
    }
}

/**
 * \brief Função para registrar posições de UEs e gNBs periodicamente
 * Formato: Time NodeType NodeID X Y Z
 */
void
PeriodicMobilityTraceWithGnbs (std::ostream *os, NodeContainer ueNodes, NodeContainer lteEnbNodes,
                                NodeContainer mmWaveEnbNodes, double interval, double endTime)
{
    double currentTime = Simulator::Now ().GetSeconds ();

    // Trace LTE eNB (sempre estático)
    for (uint32_t i = 0; i < lteEnbNodes.GetN (); ++i)
    {
        Ptr<Node> node = lteEnbNodes.Get (i);
        Ptr<MobilityModel> mobility = node->GetObject<MobilityModel> ();
        if (mobility)
        {
            Vector pos = mobility->GetPosition ();
            *os << currentTime << "\t"
                << "LTE\t"
                << node->GetId () << "\t"
                << pos.x << "\t" << pos.y << "\t" << pos.z << std::endl;
        }
    }

    // Trace gNBs mmWave (podem ser UAVs móveis)
    for (uint32_t i = 0; i < mmWaveEnbNodes.GetN (); ++i)
    {
        Ptr<Node> node = mmWaveEnbNodes.Get (i);
        Ptr<MobilityModel> mobility = node->GetObject<MobilityModel> ();
        if (mobility)
        {
            Vector pos = mobility->GetPosition ();
            *os << currentTime << "\t"
                << "UAV\t"
                << node->GetId () << "\t"
                << pos.x << "\t" << pos.y << "\t" << pos.z << std::endl;
        }
    }

    // Trace UEs
    for (uint32_t i = 0; i < ueNodes.GetN (); ++i)
    {
        Ptr<Node> node = ueNodes.Get (i);
        Ptr<MobilityModel> mobility = node->GetObject<MobilityModel> ();
        if (mobility)
        {
            Vector pos = mobility->GetPosition ();
            *os << currentTime << "\t"
                << "UE\t"
                << node->GetId () << "\t"
                << pos.x << "\t" << pos.y << "\t" << pos.z << std::endl;
        }
    }

    // Agenda próxima execução se ainda houver tempo
    if (currentTime + interval < endTime)
    {
        Simulator::Schedule (Seconds (interval), &PeriodicMobilityTraceWithGnbs, os, ueNodes,
                             lteEnbNodes, mmWaveEnbNodes, interval, endTime);
    }
}

// --- Funções de Log (do scenario-three.cc) ---
std::ofstream outFile;
std::ofstream g_pathGymOkRlfFile;
std::ofstream g_pathGymOkActivityFile;
std::ofstream g_pathGymOkPhyPowerFile;
std::map<uint16_t, std::map<uint16_t, double>> g_pathGymOkLteSinrByCell;
std::map<uint16_t, std::map<uint64_t, long double>> g_pathGymOkMmWaveSinrByCell;

void
EnsurePathGymOkFile (std::ofstream& stream, const std::string& filename, const std::string& header)
{
    if (!stream.is_open ())
    {
        stream.open (filename.c_str (), std::ios_base::out | std::ios_base::trunc);
        stream << header << std::endl;
    }
}

double
PathGymOkToDb (long double sinrLinear)
{
    if (sinrLinear <= 0.0)
    {
        return -std::numeric_limits<double>::infinity ();
    }

    return 10.0 * std::log10 (static_cast<double> (sinrLinear));
}

uint64_t
GetUnixTimestampMs (Ptr<LteEnbNetDevice> ltedev)
{
    return ltedev->GetStartTime () + Simulator::Now ().GetMilliSeconds ();
}

void
PathGymOkLteSinrTrace (std::string context,
                       uint16_t cellId,
                       uint16_t rnti,
                       double rsrp,
                       double sinr,
                       uint8_t componentCarrierId)
{
    (void) context;
    (void) rsrp;
    (void) componentCarrierId;
    g_pathGymOkLteSinrByCell[cellId][rnti] = sinr;
}

void
PathGymOkMmWaveSinrTrace (std::string context, uint64_t imsi, uint16_t cellId, long double sinr)
{
    (void) context;
    g_pathGymOkMmWaveSinrByCell[cellId][imsi] = sinr;
}

void
EmitPathGymOkRlfDump (double nowSeconds, uint16_t cellId, uint32_t badSinrUes)
{
    if (badSinrUes > 0)
    {
        std::cout << "RLF_DUMP," << nowSeconds << "," << cellId << "," << badSinrUes
                  << std::endl;
    }
}

void
PathGymOkSampleMetrics (Ptr<LteEnbNetDevice> ltedev,
                        NetDeviceContainer mmWaveEnbDevs,
                        double outageThresholdDb,
                        double interval,
                        double simTime)
{
    EnsurePathGymOkFile (g_pathGymOkRlfFile,
                         "path-gym-ok-rlf-100ms.txt",
                         "Timestamp UNIX CellType CellId TotalTrackedUes BadSinrUes BadSinrPct ThresholdDb");
    EnsurePathGymOkFile (g_pathGymOkActivityFile,
                         "path-gym-ok-cell-activity-100ms.txt",
                         "Timestamp UNIX CellType CellId Active TotalActiveCells ActiveDurationMs");
    EnsurePathGymOkFile (g_pathGymOkPhyPowerFile,
                         "path-gym-ok-phy-power-100ms.txt",
                         "Timestamp UNIX CellType CellId TxPowerDbm Active");

    const double nowSeconds = Simulator::Now ().GetSeconds ();
    const uint64_t unixTimestampMs = GetUnixTimestampMs (ltedev);

    std::vector<Ptr<MmWaveEnbNetDevice>> mmWaveDevices;
    uint32_t activeMmWaveCells = 0;
    for (uint32_t i = 0; i < mmWaveEnbDevs.GetN (); ++i)
    {
        Ptr<MmWaveEnbNetDevice> mmdev = DynamicCast<MmWaveEnbNetDevice> (mmWaveEnbDevs.Get (i));
        if (!mmdev)
        {
            continue;
        }

        mmWaveDevices.push_back (mmdev);
        if (mmdev->GetBsState ())
        {
            ++activeMmWaveCells;
        }
    }

    const uint32_t activeCellsTotal = activeMmWaveCells + 1;
    const uint16_t lteCellId = ltedev->GetCellId ();
    uint32_t lteTrackedUes = 0;
    uint32_t lteBadSinrUes = 0;
    auto lteIt = g_pathGymOkLteSinrByCell.find (lteCellId);
    if (lteIt != g_pathGymOkLteSinrByCell.end ())
    {
        lteTrackedUes = lteIt->second.size ();
        for (const auto& entry : lteIt->second)
        {
            if (PathGymOkToDb (entry.second) < outageThresholdDb)
            {
                ++lteBadSinrUes;
            }
        }
    }

    const double lteBadSinrPct =
        lteTrackedUes > 0 ? 100.0 * static_cast<double> (lteBadSinrUes) / lteTrackedUes : 0.0;
    g_pathGymOkRlfFile << nowSeconds << " " << unixTimestampMs << " LTE " << lteCellId << " "
                       << lteTrackedUes << " " << lteBadSinrUes << " " << lteBadSinrPct << " "
                       << outageThresholdDb << std::endl;
    EmitPathGymOkRlfDump (nowSeconds, lteCellId, lteBadSinrUes);
    g_pathGymOkActivityFile << nowSeconds << " " << unixTimestampMs << " LTE " << lteCellId
                            << " 1 " << activeCellsTotal << " "
                            << Simulator::Now ().GetMilliSeconds () << std::endl;

    Ptr<LteEnbPhy> ltePhy = ltedev->GetPhy ();
    g_pathGymOkPhyPowerFile << nowSeconds << " " << unixTimestampMs << " LTE " << lteCellId
                            << " " << (ltePhy ? ltePhy->GetTxPower () : 0.0) << " 1"
                            << std::endl;

    for (const auto& mmdev : mmWaveDevices)
    {
        const uint16_t mmWaveCellId = mmdev->GetCellId ();
        uint32_t trackedUes = 0;
        uint32_t badSinrUes = 0;

        auto mmIt = g_pathGymOkMmWaveSinrByCell.find (mmWaveCellId);
        if (mmIt != g_pathGymOkMmWaveSinrByCell.end ())
        {
            trackedUes = mmIt->second.size ();
            for (const auto& entry : mmIt->second)
            {
                if (PathGymOkToDb (entry.second) < outageThresholdDb)
                {
                    ++badSinrUes;
                }
            }
        }

        const double badSinrPct =
            trackedUes > 0 ? 100.0 * static_cast<double> (badSinrUes) / trackedUes : 0.0;
        const uint32_t active = mmdev->GetBsState () ? 1u : 0u;

        g_pathGymOkRlfFile << nowSeconds << " " << unixTimestampMs << " NR " << mmWaveCellId
                           << " " << trackedUes << " " << badSinrUes << " " << badSinrPct
                           << " " << outageThresholdDb << std::endl;
        EmitPathGymOkRlfDump (nowSeconds, mmWaveCellId, badSinrUes);
        g_pathGymOkActivityFile
            << nowSeconds << " " << unixTimestampMs << " NR " << mmWaveCellId << " " << active
            << " " << activeCellsTotal << " "
            << static_cast<uint64_t> (
                   std::llround (mmdev->GetAccumulatedActiveTime ().GetMilliSeconds ()))
            << std::endl;

        Ptr<MmWaveEnbPhy> mmWavePhy = mmdev->GetPhy ();
        g_pathGymOkPhyPowerFile << nowSeconds << " " << unixTimestampMs << " NR "
                                << mmWaveCellId << " "
                                << (mmWavePhy ? mmWavePhy->GetTxPower () : 0.0) << " "
                                << active << std::endl;
    }

    if (nowSeconds + interval <= simTime + 1e-6)
    {
        Simulator::Schedule (Seconds (interval),
                             &PathGymOkSampleMetrics,
                             ltedev,
                             mmWaveEnbDevs,
                             outageThresholdDb,
                             interval,
                             simTime);
    }
}

void
BsStateTrace (std::string filename, Ptr<LteEnbNetDevice> ltedev, Ptr<LteEnbRrc> lte_rrc )
{
    if (!outFile.is_open ())
    {
        outFile.open (filename.c_str (), std::ios_base::out | std::ios_base::trunc);
        NS_LOG_LOGIC ("File opened");
        outFile << "Timestamp"
        << " "
        << "UNIX"
        << " "
        << "Id"
        << " "
        << "State" << std::endl;
    }
    // Lê o mapa que indica se handover é permitido para cada célula secundária
    std::map<uint16_t, bool> entry = lte_rrc->GetAllowHandoverTo();
    for (auto it = entry.begin(); it != entry.end(); it++)
    {
        // Calcula o timestamp UNIX em milissegundos
        uint64_t unix_timestamp_ms = ltedev->GetStartTime() + Simulator::Now ().GetMilliSeconds ();
        // Escreve: Tempo Simulação (s), Timestamp UNIX (ms), CellID, Estado (1=Permitido/ON, 0=Não Permitido/OFF)
        outFile << Simulator::Now ().GetSeconds () << " " << unix_timestamp_ms << " "
        << it->first << " " << it->second << std::endl;
    }
}

// Função para imprimir a posição das UEs para gnuplot
void
PrintGnuplottableUeListToFile (std::string filename)
{
    std::ofstream outFile;
    outFile.open (filename.c_str (), std::ios_base::out | std::ios_base::trunc);
    if (!outFile.is_open ())
    {
        NS_LOG_ERROR ("Can't open file " << filename);
        return;
    }
    for (NodeList::Iterator it = NodeList::Begin (); it != NodeList::End (); ++it)
    {
        Ptr<Node> node = *it;
        int nDevs = node->GetNDevices ();
        for (int j = 0; j < nDevs; j++)
        {
            Ptr<McUeNetDevice> mcuedev = node->GetDevice (j)->GetObject<McUeNetDevice> ();
            if (mcuedev)
            {
                Vector pos = node->GetObject<MobilityModel> ()->GetPosition ();
                outFile << "set label \"" << mcuedev->GetImsi () << "\" at " << pos.x << "," << pos.y
                << " left font \"Helvetica,8\" textcolor rgb \"black\" front point pt 1 ps "
                "0.3 lc rgb \"black\" offset 0,0"
                << std::endl;
            }
        }
    }
}

// Função para imprimir a posição das eNBs/gNBs para gnuplot
void
PrintGnuplottableEnbListToFile (std::string filename)
{
    std::ofstream outFile;
    outFile.open (filename.c_str (), std::ios_base::out | std::ios_base::trunc);
    if (!outFile.is_open ())
    {
        NS_LOG_ERROR ("Can't open file " << filename);
        return;
    }
    for (NodeList::Iterator it = NodeList::Begin (); it != NodeList::End (); ++it)
    {
        Ptr<Node> node = *it;
        int nDevs = node->GetNDevices ();
        for (int j = 0; j < nDevs; j++)
        {
            Ptr<LteEnbNetDevice> enbdev = node->GetDevice (j)->GetObject<LteEnbNetDevice> ();
            Ptr<MmWaveEnbNetDevice> mmdev = node->GetDevice (j)->GetObject<MmWaveEnbNetDevice> ();
            if (enbdev) // eNB LTE (azul)
            {
                Vector pos = node->GetObject<MobilityModel> ()->GetPosition ();
                outFile << "set label \"" << enbdev->GetCellId () << "\" at " << pos.x << "," << pos.y
                << " left font \"Helvetica,8\" textcolor rgb \"blue\" front  point pt 4 ps "
                "0.3 lc rgb \"blue\" offset 0,0"
                << std::endl;
            }
            else if (mmdev) // gNB mmWave (vermelho)
            {
                Vector pos = node->GetObject<MobilityModel> ()->GetPosition ();
                outFile << "set label \"" << mmdev->GetCellId () << "\" at " << pos.x << "," << pos.y
                << " left font \"Helvetica,8\" textcolor rgb \"red\" front  point pt 4 ps "
                "0.3 lc rgb \"red\" offset 0,0"
                << std::endl;
            }
        }
    }
}

// --- Parâmetros Globais (mesclados de ambos os cenários) ---

// Parâmetros Comuns
static ns3::GlobalValue g_simTime ("simTime", "Simulation time in seconds", ns3::DoubleValue (10.0), ns3::MakeDoubleChecker<double> (0.1, 1000.0));
static ns3::GlobalValue g_ues ("ues", "Number of UEs for each mmWave ENB.", ns3::UintegerValue (9), ns3::MakeUintegerChecker<uint8_t> ()); // Ajustado para corresponder ao seu log
static ns3::GlobalValue g_indicationPeriodicity ("indicationPeriodicity", "E2 Indication Periodicity reports (value in seconds)", ns3::DoubleValue (0.1), ns3::MakeDoubleChecker<double> (0.01, 2.0));
static ns3::GlobalValue g_configuration ("configuration", "Set the wanted configuration to emulate [0=LTE 850MHz, 1=5G FR1 3.5GHz, 2=5G FR2 28GHz]", ns3::UintegerValue (1), ns3::MakeUintegerChecker<uint8_t> ()); // O-RAN 5G FR1 por padrão
static ns3::GlobalValue g_trafficModel ("trafficModel", "Type of the traffic model [0,3]", ns3::UintegerValue (3), ns3::MakeUintegerChecker<uint8_t> ()); // Ajustado para corresponder ao seu log
static ns3::GlobalValue q_useSemaphores ("useSemaphores", "If true, enables the use of semaphores for external environment control", ns3::BooleanValue (false), ns3::MakeBooleanChecker ()); // Ajustado para corresponder ao seu log de erro
static ns3::GlobalValue g_controlFileName ("controlFileName", "The path to the control file for hierarchical actions", ns3::StringValue (""), ns3::MakeStringChecker ()); // Definido para o esperado
static ns3::GlobalValue g_scheduleControlMessages (
    "scheduleControlMessages",
    "If true, execute control actions at the timestamp encoded in the control file",
    ns3::BooleanValue (false),
    ns3::MakeBooleanChecker ());
//hierarchical_actions.csv
// Parâmetros de Handover (do scenario-one)
static ns3::GlobalValue g_hoSinrDifference ("hoSinrDifference", "The SINR value difference for which a handover is triggered", ns3::DoubleValue (3), ns3::MakeDoubleChecker<double> ());

// Parâmetros de Mobilidade (do scenario-three)
static ns3::GlobalValue g_positionAllocator ("positionAllocator", "UE position allocator type [0=uniform disc, 1=around BSs, 2=Shanghai urban mobility]", ns3::UintegerValue (0), ns3::MakeUintegerChecker<uint8_t> ());
static ns3::GlobalValue g_nBsNoUesAlloc ("nBsNoUesAlloc", "Number of BS without initial UEs allocated", ns3::IntegerValue (-1), ns3::MakeIntegerChecker<int8_t> ());
static ns3::GlobalValue g_minSpeed ("minSpeed", "minimum UE speed in m/s", ns3::DoubleValue (2.0), ns3::MakeDoubleChecker<double> ());
static ns3::GlobalValue g_maxSpeed ("maxSpeed", "maximum UE speed in m/s", ns3::DoubleValue (4.0), ns3::MakeDoubleChecker<double> ());

// Parâmetros de Heurística Energy Saving (do scenario-three)
static ns3::GlobalValue g_heuristicType (
    "heuristicType",
    "Type of heuristic for managing BS status: -1=No heuristic (RL control), 0=Always ON, 1=Dynamic sleeping",
    ns3::IntegerValue (-1), ns3::MakeIntegerChecker<int8_t> ());
static ns3::GlobalValue g_sinrTh (
    "sinrTh",
    "SINR threshold for static and dynamic sleeping heuristic",
    ns3::DoubleValue (73.0), ns3::MakeDoubleChecker<double> ());
static ns3::GlobalValue g_bsOn (
    "bsOn",
    "number of BS to turn ON for static and dynamic sleeping heuristic",
    ns3::UintegerValue (2), ns3::MakeUintegerChecker<uint8_t> ());
static ns3::GlobalValue g_bsIdle (
    "bsIdle",
    "number of BS to turn IDLE for static and dynamic sleeping heuristic",
    ns3::UintegerValue (2), ns3::MakeUintegerChecker<uint8_t> ());
static ns3::GlobalValue g_bsSleep (
    "bsSleep",
    "number of BS to turn Sleep for static and dynamic sleeping heuristic",
    ns3::UintegerValue (2), ns3::MakeUintegerChecker<uint8_t> ());
static ns3::GlobalValue g_bsOff (
    "bsOff",
    "number of BS to turn Off for static and dynamic sleeping heuristic",
    ns3::UintegerValue (1), ns3::MakeUintegerChecker<uint8_t> ());

// Parâmetros Técnicos (comuns ou de um dos cenários)
static ns3::GlobalValue g_bufferSize ("bufferSize", "RLC tx buffer size (MB)", ns3::UintegerValue (1), ns3::MakeUintegerChecker<uint32_t> ());
static ns3::GlobalValue g_rlcAmEnabled ("rlcAmEnabled", "If true, use RLC AM, else use RLC UM", ns3::BooleanValue (true), ns3::MakeBooleanChecker ());
static ns3::GlobalValue g_enableTraces ("enableTraces", "If true, generate ns-3 traces", ns3::BooleanValue (true), ns3::MakeBooleanChecker ());
static ns3::GlobalValue g_e2lteEnabled ("e2lteEnabled", "If true, send LTE E2 reports", ns3::BooleanValue (true), ns3::MakeBooleanChecker ());
static ns3::GlobalValue g_e2nrEnabled ("e2nrEnabled", "If true, send NR E2 reports", ns3::BooleanValue (true), ns3::MakeBooleanChecker ());
static ns3::GlobalValue g_e2du ("e2du", "If true, send DU reports", ns3::BooleanValue (true), ns3::MakeBooleanChecker ());
static ns3::GlobalValue g_e2cuUp ("e2cuUp", "If true, send CU-UP reports", ns3::BooleanValue (true), ns3::MakeBooleanChecker ());
static ns3::GlobalValue g_e2cuCp ("e2cuCp", "If true, send CU-CP reports", ns3::BooleanValue (true), ns3::MakeBooleanChecker ());
static ns3::GlobalValue g_dataRate ("dataRate", "Set the data rate to be used [0=low, 1=high]", ns3::DoubleValue (0), ns3::MakeDoubleChecker<double> (0, 1));
static ns3::GlobalValue g_reducedPmValues ("reducedPmValues", "If true, use a subset of the pm containers", ns3::BooleanValue (false), ns3::MakeBooleanChecker ()); // Ajustado para corresponder ao seu log
static ns3::GlobalValue g_outageThreshold ("outageThreshold", "SNR threshold for outage events [dB]", ns3::DoubleValue (-5.0), ns3::MakeDoubleChecker<double> ()); // Ajustado para corresponder ao seu log
static ns3::GlobalValue g_basicCellId ("basicCellId", "The next value will be the first cellId", ns3::UintegerValue (1), ns3::MakeUintegerChecker<uint8_t> ());
static ns3::GlobalValue g_numberOfRaPreambles ("numberOfRaPreambles", "Number of RA preambles", ns3::UintegerValue (40), ns3::MakeUintegerChecker<uint8_t> ()); // Ajustado para corresponder ao seu log
static ns3::GlobalValue g_handoverMode ("handoverMode", "HO euristic to be used", ns3::StringValue ("DynamicTtt"), ns3::MakeStringChecker ()); // Ajustado para corresponder ao seu log
static ns3::GlobalValue g_e2TermIp ("e2TermIp", "The IP address of the RIC E2 termination", ns3::StringValue ("127.0.0.1"), ns3::MakeStringChecker ());
static ns3::GlobalValue g_enableE2FileLogging ("enableE2FileLogging", "If true, generate offline file logging instead of connecting to RIC", ns3::BooleanValue (true), ns3::MakeBooleanChecker ());
static ns3::GlobalValue g_pathGymOkMetrics ("pathGymOkMetrics", "If true, enable reversible path-gym-ok 100 ms KPI hooks", ns3::BooleanValue (true), ns3::MakeBooleanChecker ());

// Adicionado RngRun para corresponder ao seu log
static ns3::GlobalValue g_rngRun ("RngRun", "Seed for random number generation", ns3::UintegerValue (555), ns3::MakeUintegerChecker<uint32_t> ());

// --- UAV Mobility Parameters ---
static ns3::GlobalValue g_uavMobilityMode ("uavMobilityMode", "UAV mobility mode [0=static, 1=mobile]", ns3::UintegerValue (0), ns3::MakeUintegerChecker<uint8_t> (0, 1));
static ns3::GlobalValue g_uavFlightPattern ("uavFlightPattern", "UAV flight pattern [0=hovering, 1=circular, 2=patrol, 3=figure8, 4=grid, 5=altitude]", ns3::UintegerValue (0), ns3::MakeUintegerChecker<uint8_t> (0, 5));
static ns3::GlobalValue g_uavBaseAltitude ("uavBaseAltitude", "Base altitude for UAV gNBs in meters (30-300m)", ns3::DoubleValue (100.0), ns3::MakeDoubleChecker<double> (30.0, 300.0));
static ns3::GlobalValue g_uavAltitudeVariation ("uavAltitudeVariation", "Maximum altitude variation for UAVs in meters", ns3::DoubleValue (30.0), ns3::MakeDoubleChecker<double> (0.0, 100.0));
static ns3::GlobalValue g_uavMaxSpeed ("uavMaxSpeed", "Maximum UAV speed in m/s (1-30 m/s)", ns3::DoubleValue (15.0), ns3::MakeDoubleChecker<double> (1.0, 30.0));
static ns3::GlobalValue g_uavOrbitRadius ("uavOrbitRadius", "Radius for circular orbit pattern in meters", ns3::DoubleValue (200.0), ns3::MakeDoubleChecker<double> (50.0, 500.0));
static ns3::GlobalValue g_uavPatrolLength ("uavPatrolLength", "Length for patrol pattern in meters", ns3::DoubleValue (400.0), ns3::MakeDoubleChecker<double> (100.0, 1000.0));
static ns3::GlobalValue g_uavGridSize ("uavGridSize", "Grid size for coverage pattern in meters", ns3::DoubleValue (300.0), ns3::MakeDoubleChecker<double> (100.0, 500.0));
static ns3::GlobalValue g_enableGnbMobilityTrace ("enableGnbMobilityTrace", "If true, include gNB positions in mobility trace", ns3::BooleanValue (true), ns3::MakeBooleanChecker ());

int
main (int argc, char *argv[])
{
    // Configura o nível de log
    // LogComponentEnableAll (LOG_PREFIX_ALL); // Descomente para log máximo
    LogComponentEnable ("ScenarioHierarchical", LOG_LEVEL_INFO);
    // LogComponentEnable ("LteEnbNetDevice", LOG_LEVEL_DEBUG); // Descomente para depurar leitura do ficheiro
    // LogComponentEnable ("LteEnbRrc", LOG_LEVEL_DEBUG); // Descomente para depurar HO e estado

    // Define os limites do cenário
    double maxXAxis = 4000;
    double maxYAxis = 4000;

    // Processa argumentos da linha de comando
    CommandLine cmd;
    cmd.Parse (argc, argv);

    // --- Leitura e Configuração de Parâmetros ---
    bool harqEnabled = true;
    UintegerValue uintegerValue;
    IntegerValue integerValue;
    BooleanValue booleanValue;
    StringValue stringValue;
    DoubleValue doubleValue;

    // Lê os valores dos parâmetros globais para variáveis locais
    GlobalValue::GetValueByName ("hoSinrDifference", doubleValue);
    double hoSinrDifference = doubleValue.Get ();
    GlobalValue::GetValueByName ("dataRate", doubleValue);
    double dataRateFromConf = doubleValue.Get ();
    GlobalValue::GetValueByName ("rlcAmEnabled", booleanValue);
    bool rlcAmEnabled = booleanValue.Get ();
    GlobalValue::GetValueByName ("bufferSize", uintegerValue);
    uint32_t bufferSize = uintegerValue.Get ();
    GlobalValue::GetValueByName ("basicCellId", uintegerValue);
    uint16_t basicCellId = uintegerValue.Get ();
    (void) basicCellId; // Evita aviso de não utilizado
    GlobalValue::GetValueByName ("enableTraces", booleanValue);
    bool enableTraces = booleanValue.Get ();
    GlobalValue::GetValueByName ("trafficModel", uintegerValue);
    uint8_t trafficModel = uintegerValue.Get ();
    (void) trafficModel; // Evita aviso de não utilizado
    GlobalValue::GetValueByName ("nBsNoUesAlloc", integerValue);
    int8_t nBsNoUesAlloc = integerValue.Get ();
    (void) nBsNoUesAlloc; // Evita aviso de não utilizado
    GlobalValue::GetValueByName ("positionAllocator", uintegerValue);
    uint8_t positionAllocator = uintegerValue.Get ();
    GlobalValue::GetValueByName ("outageThreshold",doubleValue);
    double outageThreshold = doubleValue.Get ();
    GlobalValue::GetValueByName ("handoverMode", stringValue);
    std::string handoverMode = stringValue.Get ();
    GlobalValue::GetValueByName ("minSpeed", doubleValue);
    double minSpeed = doubleValue.Get ();
    GlobalValue::GetValueByName ("maxSpeed", doubleValue);
    double maxSpeed = doubleValue.Get ();
    GlobalValue::GetValueByName ("indicationPeriodicity", doubleValue);
    double indicationPeriodicity = doubleValue.Get ();
    GlobalValue::GetValueByName ("useSemaphores", booleanValue);
    bool useSemaphores = booleanValue.Get ();
    GlobalValue::GetValueByName ("controlFileName", stringValue);
    std::string controlFilename = stringValue.Get ();
    GlobalValue::GetValueByName ("scheduleControlMessages", booleanValue);
    bool scheduleControlMessages = booleanValue.Get ();


    //std::cout << "### controlFilename: " << controlFilename << "### " << std::endl;

    if (controlFilename == "none"){
        std::cout << "No control file provided, running with default settings." << std::endl;
        controlFilename = "";
    }

    // Heuristic Type for Energy Saving
    GlobalValue::GetValueByName ("heuristicType", integerValue);
    int8_t heuristicType = integerValue.Get ();
    GlobalValue::GetValueByName ("sinrTh", doubleValue);
    double sinrTh = doubleValue.Get ();
    GlobalValue::GetValueByName ("bsOn", uintegerValue);
    int bsOn = uintegerValue.Get ();
    GlobalValue::GetValueByName ("bsIdle", uintegerValue);
    int bsIdle = uintegerValue.Get ();
    GlobalValue::GetValueByName ("bsSleep", uintegerValue);
    int bsSleep = uintegerValue.Get ();
    GlobalValue::GetValueByName ("bsOff", uintegerValue);
    int bsOff = uintegerValue.Get ();

    // E2 Logging settings
    GlobalValue::GetValueByName ("e2lteEnabled", booleanValue);
    bool e2lteEnabled = booleanValue.Get ();
    GlobalValue::GetValueByName ("e2nrEnabled", booleanValue);
    bool e2nrEnabled = booleanValue.Get ();
    GlobalValue::GetValueByName ("e2du", booleanValue);
    bool e2du = booleanValue.Get ();
    GlobalValue::GetValueByName ("e2cuUp", booleanValue);
    bool e2cuUp = booleanValue.Get ();
    GlobalValue::GetValueByName ("e2cuCp", booleanValue);
    bool e2cuCp = booleanValue.Get ();
    GlobalValue::GetValueByName ("reducedPmValues", booleanValue);
    bool reducedPmValues = booleanValue.Get ();
    GlobalValue::GetValueByName ("enableE2FileLogging", booleanValue);
    bool enableE2FileLogging = booleanValue.Get ();
    GlobalValue::GetValueByName ("pathGymOkMetrics", booleanValue);
    bool pathGymOkMetrics = booleanValue.Get ();

    // --- CORREÇÃO: Adiciona leituras em falta ---
    GlobalValue::GetValueByName ("numberOfRaPreambles", uintegerValue);
    uint8_t numberOfRaPreambles = uintegerValue.Get ();
    GlobalValue::GetValueByName ("e2TermIp", stringValue);
    std::string e2TermIp = stringValue.Get ();
    // --- FIM DA CORREÇÃO ---

    // --- Leitura dos parâmetros UAV ---
    GlobalValue::GetValueByName ("uavMobilityMode", uintegerValue);
    uint8_t uavMobilityMode = uintegerValue.Get ();
    GlobalValue::GetValueByName ("uavFlightPattern", uintegerValue);
    uint8_t uavFlightPattern = uintegerValue.Get ();
    GlobalValue::GetValueByName ("uavBaseAltitude", doubleValue);
    double uavBaseAltitude = doubleValue.Get ();
    GlobalValue::GetValueByName ("uavAltitudeVariation", doubleValue);
    double uavAltitudeVariation = doubleValue.Get ();
    GlobalValue::GetValueByName ("uavMaxSpeed", doubleValue);
    double uavMaxSpeed = doubleValue.Get ();
    GlobalValue::GetValueByName ("uavOrbitRadius", doubleValue);
    double uavOrbitRadius = doubleValue.Get ();
    GlobalValue::GetValueByName ("uavPatrolLength", doubleValue);
    double uavPatrolLength = doubleValue.Get ();
    GlobalValue::GetValueByName ("uavGridSize", doubleValue);
    double uavGridSize = doubleValue.Get ();
    GlobalValue::GetValueByName ("enableGnbMobilityTrace", booleanValue);
    bool enableGnbMobilityTrace = booleanValue.Get ();

    NS_LOG_INFO ("UAV Parameters: mode=" << (int)uavMobilityMode << ", pattern=" << (int)uavFlightPattern
                 << ", altitude=" << uavBaseAltitude << "m, speed=" << uavMaxSpeed << "m/s");
    // --- FIM dos parâmetros UAV ---

    // Aplica a seed RngRun
    GlobalValue::GetValueByName("RngRun", uintegerValue);
    RngSeedManager::SetRun (uintegerValue.Get ());

    GlobalValue::GetValueByName ("simTime", doubleValue);
    double simTimeLoaded = doubleValue.Get ();
    GlobalValue::GetValueByName ("configuration", uintegerValue);
    uint8_t configurationLoaded = uintegerValue.Get ();

    auto boolStr = [] (bool value) -> const char*
    {
        return value ? "true" : "false";
    };

    std::string configurationLabel;
    switch (configurationLoaded)
    {
        case 0: configurationLabel = "LTE 850MHz"; break;
        case 1: configurationLabel = "5G FR1 3.5GHz"; break;
        case 2: configurationLabel = "5G FR2 28GHz"; break;
        default: configurationLabel = "unknown"; break;
    }

    std::string trafficModelLabel;
    switch (trafficModel)
    {
        case 0: trafficModelLabel = "full-buffer UDP"; break;
        case 1: trafficModelLabel = "half full-buffer / half bursty"; break;
        case 2: trafficModelLabel = "all bursty mixed TCP/UDP"; break;
        case 3: trafficModelLabel = "4-class heterogeneous mix"; break;
        default: trafficModelLabel = "unknown"; break;
    }

    std::string positionAllocatorLabel;
    switch (positionAllocator)
    {
        case 0: positionAllocatorLabel = "uniform disc"; break;
        case 1: positionAllocatorLabel = "around BSs"; break;
        case 2: positionAllocatorLabel = "Shanghai urban mobility"; break;
        default: positionAllocatorLabel = "unknown"; break;
    }

    std::string heuristicLabel;
    switch (heuristicType)
    {
        case -1: heuristicLabel = "external/RL control"; break;
        case 0: heuristicLabel = "always on"; break;
        case 1: heuristicLabel = "dynamic sleeping"; break;
        default: heuristicLabel = "unknown"; break;
    }

    std::string handoverModeLabel = handoverMode;

    std::string resolvedDataRate;
    switch (configurationLoaded)
    {
        case 0:
            resolvedDataRate = (dataRateFromConf == 0 ? "1.5Mbps" : "4.5Mbps");
            break;
        case 1:
            resolvedDataRate = (dataRateFromConf == 0 ? "50Mbps" : "150Mbps");
            break;
        case 2:
            resolvedDataRate = (dataRateFromConf == 0 ? "15Mbps" : "45Mbps");
            break;
        default:
            resolvedDataRate = "unknown";
            break;
    }

    std::string uavFlightPatternLabel;
    switch (uavFlightPattern)
    {
        case 0: uavFlightPatternLabel = "hovering"; break;
        case 1: uavFlightPatternLabel = "circular"; break;
        case 2: uavFlightPatternLabel = "patrol"; break;
        case 3: uavFlightPatternLabel = "figure8"; break;
        case 4: uavFlightPatternLabel = "grid"; break;
        case 5: uavFlightPatternLabel = "altitude"; break;
        default: uavFlightPatternLabel = "unknown"; break;
    }

    std::cout << "\n======================================================\n";
    std::cout << "--- SIMULATION PARAMETERS LOADED (SEM VERIFICATION) ---\n";
    std::cout << "simTime:                 " << simTimeLoaded << " s\n";
    std::cout << "RngRun:                  " << uintegerValue.Get() << "\n";
    std::cout << "configuration:           " << (int)configurationLoaded << " (" << configurationLabel << ")\n";
    std::cout << "trafficModel:            " << (int)trafficModel << " (" << trafficModelLabel << ")\n";
    std::cout << "dataRateProfile:         " << dataRateFromConf << " (resolved " << resolvedDataRate << ")\n";
    std::cout << "rlcAmEnabled:            " << boolStr(rlcAmEnabled) << "\n";
    std::cout << "bufferSize:              " << bufferSize << " MB\n";
    std::cout << "basicCellId:             " << basicCellId << "\n";
    std::cout << "controlFileName:         " << controlFilename << "\n";
    std::cout << "useSemaphores:           " << boolStr(useSemaphores) << "\n";
    std::cout << "scheduleControlMessages: " << boolStr(scheduleControlMessages) << "\n";
    std::cout << "positionAllocator:       " << (int)positionAllocator << " (" << positionAllocatorLabel << ")\n";
    std::cout << "nBsNoUesAlloc:           " << (int)nBsNoUesAlloc << "\n";
    std::cout << "minSpeed/maxSpeed:       " << minSpeed << " / " << maxSpeed << " m/s\n";
    std::cout << "handoverMode:            " << handoverModeLabel << "\n";
    std::cout << "outageThreshold:         " << outageThreshold << " dB\n";
    std::cout << "indicationPeriodicity:   " << indicationPeriodicity << " s\n";
    std::cout << "numberOfRaPreambles:     " << (int)numberOfRaPreambles << "\n";
    std::cout << "enableTraces:            " << boolStr(enableTraces) << "\n";
    std::cout << "reducedPmValues:         " << boolStr(reducedPmValues) << "\n";
    std::cout << "enableE2FileLogging:     " << boolStr(enableE2FileLogging) << "\n";
    std::cout << "pathGymOkMetrics:        " << boolStr(pathGymOkMetrics) << "\n";
    std::cout << "e2lte/e2nr:              " << boolStr(e2lteEnabled) << " / " << boolStr(e2nrEnabled) << "\n";
    std::cout << "e2du/e2cuUp/e2cuCp:      " << boolStr(e2du) << " / " << boolStr(e2cuUp) << " / " << boolStr(e2cuCp) << "\n";
    std::cout << "e2TermIp:                " << e2TermIp << "\n";
    std::cout << "heuristicType:           " << (int)heuristicType << " (" << heuristicLabel << ")\n";
    std::cout << "sinrTh:                  " << sinrTh << "\n";
    std::cout << "bsOn/bsIdle/bsSleep/bsOff: "
              << bsOn << " / " << bsIdle << " / " << bsSleep << " / " << bsOff << "\n";
    std::cout << "uavMobilityMode:         " << (int)uavMobilityMode << " (" << (uavMobilityMode == 0 ? "static" : "mobile") << ")\n";
    std::cout << "uavFlightPattern:        " << (int)uavFlightPattern << " (" << uavFlightPatternLabel << ")\n";
    std::cout << "uavBaseAltitude:         " << uavBaseAltitude << " m\n";
    std::cout << "uavAltitudeVariation:    " << uavAltitudeVariation << " m\n";
    std::cout << "uavMaxSpeed:             " << uavMaxSpeed << " m/s\n";
    std::cout << "uavOrbitRadius:          " << uavOrbitRadius << " m\n";
    std::cout << "uavPatrolLength:         " << uavPatrolLength << " m\n";
    std::cout << "uavGridSize:             " << uavGridSize << " m\n";
    std::cout << "enableGnbMobilityTrace:  " << boolStr(enableGnbMobilityTrace) << "\n";
    std::cout << "======================================================\n\n";


    // --- Configurações Padrão do ns-3 (mescladas e corrigidas) ---
    Config::SetDefault ("ns3::LteEnbNetDevice::UseSemaphores", BooleanValue (useSemaphores));
    Config::SetDefault ("ns3::LteEnbNetDevice::ControlFileName", StringValue(controlFilename));
    Config::SetDefault ("ns3::LteEnbNetDevice::ScheduleControlMessages",
                        BooleanValue (scheduleControlMessages));
    Config::SetDefault ("ns3::LteEnbNetDevice::E2Periodicity", DoubleValue (indicationPeriodicity));
    Config::SetDefault ("ns3::MmWaveEnbNetDevice::E2Periodicity", DoubleValue (indicationPeriodicity));

    // Configuração E2
    Config::SetDefault ("ns3::MmWaveHelper::E2ModeLte", BooleanValue(e2lteEnabled));
    Config::SetDefault ("ns3::MmWaveHelper::E2ModeNr", BooleanValue(e2nrEnabled));
    Config::SetDefault ("ns3::MmWaveHelper::E2Periodicity", DoubleValue (indicationPeriodicity));
    Config::SetDefault ("ns3::MmWaveEnbNetDevice::EnableDuReport", BooleanValue(e2du));
    Config::SetDefault ("ns3::MmWaveEnbNetDevice::EnableCuUpReport", BooleanValue(e2cuUp));
    Config::SetDefault ("ns3::LteEnbNetDevice::EnableCuUpReport", BooleanValue(e2cuUp));
    Config::SetDefault ("ns3::MmWaveEnbNetDevice::EnableCuCpReport", BooleanValue(e2cuCp));
    Config::SetDefault ("ns3::LteEnbNetDevice::EnableCuCpReport", BooleanValue(e2cuCp));
    Config::SetDefault ("ns3::MmWaveEnbNetDevice::ReducedPmValues", BooleanValue (reducedPmValues));
    Config::SetDefault ("ns3::LteEnbNetDevice::ReducedPmValues", BooleanValue (reducedPmValues));
    Config::SetDefault ("ns3::LteEnbNetDevice::EnableE2FileLogging", BooleanValue (enableE2FileLogging));
    Config::SetDefault ("ns3::MmWaveEnbNetDevice::EnableE2FileLogging", BooleanValue (enableE2FileLogging));
    if (pathGymOkMetrics)
    {
        Config::SetDefault ("ns3::RadioBearerStatsCalculator::EpochDuration",
                            TimeValue (Seconds (indicationPeriodicity)));
        Config::SetDefault ("ns3::MmWaveBearerStatsCalculator::EpochDuration",
                            TimeValue (Seconds (indicationPeriodicity)));
    }

    // Configuração RRC e Handover
    Config::SetDefault ("ns3::LteEnbRrc::OutageThreshold", DoubleValue (outageThreshold));
    Config::SetDefault ("ns3::LteEnbRrc::SecondaryCellHandoverMode", StringValue (handoverMode));
    Config::SetDefault ("ns3::LteEnbRrc::HoSinrDifference", DoubleValue (hoSinrDifference));

    // Outras configurações (incluindo correções HARQ e RACH)
    Config::SetDefault ("ns3::MmWaveHelper::RlcAmEnabled", BooleanValue (rlcAmEnabled));
    Config::SetDefault ("ns3::MmWaveHelper::HarqEnabled", BooleanValue (harqEnabled));
    Config::SetDefault ("ns3::MmWaveFlexTtiMacScheduler::HarqEnabled", BooleanValue (harqEnabled)); // Garante consistência
    Config::SetDefault ("ns3::MmWavePhyMacCommon::NumHarqProcess", UintegerValue (100)); // --- CORREÇÃO HARQ ---
    Config::SetDefault ("ns3::MmWaveEnbMac::NumberOfRaPreambles", UintegerValue (numberOfRaPreambles)); // Usa variável lida
    Config::SetDefault ("ns3::MmWaveHelper::UseIdealRrc", BooleanValue (true)); // Comum nos outros
    Config::SetDefault ("ns3::MmWaveHelper::BasicCellId", UintegerValue (basicCellId)); // Usa variável lida
    Config::SetDefault ("ns3::MmWaveHelper::BasicImsi", UintegerValue ((basicCellId-1))); // Usa variável lida
    Config::SetDefault ("ns3::MmWaveHelper::E2TermIp", StringValue (e2TermIp)); // Usa variável lida
    Config::SetDefault ("ns3::ThreeGppChannelModel::UpdatePeriod", TimeValue (MilliSeconds (100.0))); // Comum nos outros
    Config::SetDefault ("ns3::ThreeGppChannelConditionModel::UpdatePeriod", TimeValue (MilliSeconds (100))); // Comum nos outros
    Config::SetDefault ("ns3::LteRlcAm::ReportBufferStatusTimer", TimeValue (MilliSeconds (10.0))); // Comum nos outros
    Config::SetDefault ("ns3::LteRlcUmLowLat::ReportBufferStatusTimer", TimeValue (MilliSeconds (10.0))); // Comum nos outros
    Config::SetDefault ("ns3::LteRlcUm::MaxTxBufferSize", UintegerValue (bufferSize * 1024 * 1024));
    Config::SetDefault ("ns3::LteRlcUmLowLat::MaxTxBufferSize", UintegerValue (bufferSize * 1024 * 1024));
    Config::SetDefault ("ns3::LteRlcAm::MaxTxBufferSize", UintegerValue (bufferSize * 1024 * 1024));


    // --- Construção do Cenário (baseado no scenario-three) ---

    // Configuração de Frequência, Largura de Banda, ISD, Antenas e Data Rate
    double bandwidth;
    double centerFrequency;
    double isd;
    int numAntennasMcUe;
    (void) numAntennasMcUe; // Evita aviso
    int numAntennasMmWave;
    (void) numAntennasMmWave; // Evita aviso
    std::string dataRate;

    GlobalValue::GetValueByName ("configuration", uintegerValue);
    uint8_t configuration = uintegerValue.Get ();
    switch (configuration)
    {
        case 0:
            centerFrequency = 850e6; bandwidth = 20e6; isd = 1700;
            numAntennasMcUe = 1; numAntennasMmWave = 1;
            dataRate = (dataRateFromConf == 0 ? "1.5Mbps" : "4.5Mbps");
            break;
        case 1: // O-RAN 5G NR FR1 (n78 band) - Urban Macro
            centerFrequency = 3.5e9;    // 3.5 GHz (n78 band, O-RAN typical)
            bandwidth = 100e6;          // 100 MHz (5G NR FR1 max)
            isd = 500;                  // 500m ISD (O-RAN urban macro typical)
            numAntennasMcUe = 4;        // 4 antennas UE (5G typical)
            numAntennasMmWave = 32;     // 32 antennas gNB (Massive MIMO)
            dataRate = (dataRateFromConf == 0 ? "50Mbps" : "150Mbps"); // 5G typical rates
            break;
        case 2:
            centerFrequency = 28e9; bandwidth = 100e6; isd = 200;
            numAntennasMcUe = 16; numAntennasMmWave = 64;
            dataRate = (dataRateFromConf == 0 ? "15Mbps" : "45Mbps");
            break;
        default:
            NS_FATAL_ERROR ("Configuration not recognized" << configuration);
            break;
    }

    // Aplica configurações de BW e Frequência Central
    Config::SetDefault ("ns3::MmWavePhyMacCommon::Bandwidth", DoubleValue (bandwidth));
    Config::SetDefault ("ns3::MmWavePhyMacCommon::CenterFreq", DoubleValue (centerFrequency));

    // Cria Helpers
    Ptr<MmWaveHelper> mmwaveHelper = CreateObject<MmWaveHelper> ();
    mmwaveHelper->SetPathlossModelType ("ns3::ThreeGppUmiStreetCanyonPropagationLossModel");
    // Adiciona configuração de ChannelConditionModel (presente no scenario-three)
    mmwaveHelper->SetChannelConditionModelType ("ns3::ThreeGppUmiStreetCanyonChannelConditionModel");

    // Configura Antenas (presente no scenario-three)
    mmwaveHelper->SetUePhasedArrayModelAttribute("NumColumns", UintegerValue(std::sqrt(numAntennasMcUe)));
    mmwaveHelper->SetUePhasedArrayModelAttribute("NumRows", UintegerValue(std::sqrt(numAntennasMcUe)));
    mmwaveHelper->SetEnbPhasedArrayModelAttribute("NumColumns",UintegerValue(std::sqrt(numAntennasMmWave)));
    mmwaveHelper->SetEnbPhasedArrayModelAttribute("NumRows", UintegerValue(std::sqrt(numAntennasMmWave)));

    Ptr<MmWavePointToPointEpcHelper> epcHelper = CreateObject<MmWavePointToPointEpcHelper> ();
    mmwaveHelper->SetEpcHelper (epcHelper);

    // Define número de nós
    uint8_t nMmWaveEnbNodes = 7;
    uint8_t nLteEnbNodes = 1;
    GlobalValue::GetValueByName ("ues", uintegerValue);
    uint32_t ues_per_gnb = uintegerValue.Get (); // Renomeado para clareza
    uint8_t nUeNodes = ues_per_gnb * nMmWaveEnbNodes;

    // Cria nós EPC (PGW) e Host Remoto
    Ptr<Node> pgw = epcHelper->GetPgwNode ();
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create (1);
    Ptr<Node> remoteHost = remoteHostContainer.Get (0);
    InternetStackHelper internet;
    internet.Install (remoteHostContainer);

    // Conecta Host Remoto ao PGW
    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute ("DataRate", DataRateValue (DataRate ("100Gb/s")));
    p2ph.SetDeviceAttribute ("Mtu", UintegerValue (2500));
    // Adiciona Delay (presente no scenario-three)
    p2ph.SetChannelAttribute ("Delay", TimeValue (Seconds (0.010)));
    NetDeviceContainer internetDevices = p2ph.Install (pgw, remoteHost);
    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase ("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign (internetDevices);
    Ipv4Address remoteHostAddr = internetIpIfaces.GetAddress (1);
    (void) remoteHostAddr; // Evita aviso
    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    Ptr<Ipv4StaticRouting> remoteHostStaticRouting = ipv4RoutingHelper.GetStaticRouting (remoteHost->GetObject<Ipv4> ());
    remoteHostStaticRouting->AddNetworkRouteTo (Ipv4Address ("7.0.0.0"), Ipv4Mask ("255.0.0.0"), 1);

    // Cria nós UE e eNB/gNB
    NodeContainer ueNodes;
    NodeContainer mmWaveEnbNodes;
    NodeContainer lteEnbNodes;
    NodeContainer allEnbNodes;
    mmWaveEnbNodes.Create (nMmWaveEnbNodes);
    lteEnbNodes.Create (nLteEnbNodes);
    ueNodes.Create (nUeNodes);
    allEnbNodes.Add (lteEnbNodes);
    allEnbNodes.Add (mmWaveEnbNodes);

    // Posiciona eNBs/gNBs (Layout Hexagonal)
    // Centro do cenário - eNB LTE fica no solo (altura 3m)
    Vector centerPosition = Vector (maxXAxis / 2, maxYAxis / 2, 3);

    // Calcula posições base (X, Y) para todos os gNBs mmWave
    std::vector<Vector> mmWaveBasePositions;
    mmWaveBasePositions.push_back (Vector (centerPosition.x, centerPosition.y, 0)); // gNB central co-localizado
    for (int8_t i = 0; i < (nMmWaveEnbNodes - 1); ++i) // 6 gNBs periféricos
    {
        double x = centerPosition.x + isd * cos ((2 * M_PI * i) / (nMmWaveEnbNodes - 1));
        double y = centerPosition.y + isd * sin ((2 * M_PI * i) / (nMmWaveEnbNodes - 1));
        mmWaveBasePositions.push_back (Vector (x, y, 0));
    }

    // --- Posiciona eNB LTE (sempre estático no solo) ---
    Ptr<ListPositionAllocator> ltePositionAlloc = CreateObject<ListPositionAllocator> ();
    ltePositionAlloc->Add (centerPosition); // LTE eNB no centro, altura 3m

    MobilityHelper lteMobility;
    lteMobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
    lteMobility.SetPositionAllocator (ltePositionAlloc);
    lteMobility.Install (lteEnbNodes);
    NS_LOG_INFO ("LTE eNB positioned at ground level (3m) at center: (" << centerPosition.x << ", " << centerPosition.y << ")");

    // --- Posiciona gNBs mmWave (podem ser UAVs ou estáticos) ---
    // Lê simTime aqui para usar na configuração de mobilidade UAV
    GlobalValue::GetValueByName ("simTime", doubleValue);
    double simTime = doubleValue.Get ();

    if (uavMobilityMode == 0)
    {
        // Modo estático: gNBs mmWave em posição fixa na altitude UAV
        Ptr<ListPositionAllocator> mmWavePositionAlloc = CreateObject<ListPositionAllocator> ();
        for (const auto& basePos : mmWaveBasePositions)
        {
            mmWavePositionAlloc->Add (Vector (basePos.x, basePos.y, uavBaseAltitude));
        }

        MobilityHelper mmWaveMobility;
        mmWaveMobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
        mmWaveMobility.SetPositionAllocator (mmWavePositionAlloc);
        mmWaveMobility.Install (mmWaveEnbNodes);

        NS_LOG_INFO ("mmWave gNBs positioned as static UAVs at altitude " << uavBaseAltitude << "m");
    }
    else
    {
        // Modo móvel: gNBs mmWave como UAVs com trajetórias
        UAVFlightPattern pattern = static_cast<UAVFlightPattern> (uavFlightPattern);

        SetupUAVMobility (mmWaveEnbNodes,
                          mmWaveBasePositions,
                          centerPosition.x,
                          centerPosition.y,
                          uavBaseAltitude,
                          uavAltitudeVariation,
                          simTime,
                          pattern,
                          uavMaxSpeed,
                          uavOrbitRadius,
                          uavPatrolLength,
                          uavGridSize);

        NS_LOG_INFO ("mmWave gNBs configured as mobile UAVs with pattern " << uavFlightPattern
                     << " at base altitude " << uavBaseAltitude << "m");
    }

    // Considera trocar modelo de propagação para altitudes elevadas
    // (opcional - comentado por padrão para manter compatibilidade)
    // if (uavBaseAltitude > 30.0)
    // {
    //     mmwaveHelper->SetPathlossModelType ("ns3::ThreeGppUmaPropagationLossModel");
    //     mmwaveHelper->SetChannelConditionModelType ("ns3::ThreeGppUmaChannelConditionModel");
    //     NS_LOG_INFO ("Switched to UMa propagation model for elevated UAV altitude");
    // }

    // Posiciona UEs (Lógica de mobilidade flexível do scenario-three)
    MobilityHelper uemobility;
    Ptr<UniformRandomVariable> speedVar = CreateObject<UniformRandomVariable> (); // Renomeado para evitar conflito
    speedVar->SetAttribute ("Min", DoubleValue (minSpeed));
    speedVar->SetAttribute ("Max", DoubleValue (maxSpeed));
    // Adiciona Random Variable para tempo de mudança de direção (do scenario-three)
    Ptr<UniformRandomVariable> puntTimeDirection = CreateObject<UniformRandomVariable> ();
    puntTimeDirection->SetAttribute ("Min", DoubleValue (1));
    puntTimeDirection->SetAttribute ("Max", DoubleValue (3));
    double timeDirection=puntTimeDirection->GetValue();

    // Nota: simTime já foi lido anteriormente para a configuração UAV

    switch (positionAllocator)
    {
        case 0: { // Distribuição uniforme no disco central
            Ptr<UniformDiscPositionAllocator> uePositionAlloc = CreateObject<UniformDiscPositionAllocator> ();
            uePositionAlloc->SetX(centerPosition.x);
            uePositionAlloc->SetY(centerPosition.y);
            uePositionAlloc->SetRho(isd);
            // Usa RandomWalk2dMobilityModel com tempo (do scenario-three)
            uemobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                                        "Mode", StringValue("Time"),
                                        "Time", StringValue(std::to_string(timeDirection) + "s"),
                                        "Speed", PointerValue(speedVar),
                                        "Bounds", RectangleValue(Rectangle(0, maxXAxis, 0, maxYAxis)));
            uemobility.SetPositionAllocator(uePositionAlloc);
            uemobility.Install(ueNodes);
            break;
        }
        case 1: { // Alocação em torno de um subconjunto de BSs (lógica complexa do scenario-three)
            if (nBsNoUesAlloc == -1)
            {
                NS_FATAL_ERROR("nBsNoUesAlloc (-1) incorrecto para positionAllocator=1.");
            }
            if (nBsNoUesAlloc >= nMmWaveEnbNodes)
            {
                NS_FATAL_ERROR("nBsNoUesAlloc (" << nBsNoUesAlloc << ") maior ou igual ao número de gNBs mmWave (" << nMmWaveEnbNodes << ").");
            }

            // Usa as posições base dos gNBs mmWave já calculadas
            std::vector<Vector> bsCoords = mmWaveBasePositions;

            // Embaralha as posições para escolher aleatoriamente quais BSs não terão UEs
            std::srand(std::time(0)); // Usa time(0) para seed
            std::shuffle(bsCoords.begin(), bsCoords.end(), std::mt19937(std::random_device()()));

            NS_LOG_INFO("Alocando UEs em torno de " << (nMmWaveEnbNodes - nBsNoUesAlloc) << " gNBs mmWave.");

            // Calcula quantos UEs por gNB ativo
            uint32_t numActiveGnbs = nMmWaveEnbNodes - nBsNoUesAlloc;
            uint32_t nodeGroupSize = nUeNodes / numActiveGnbs;
            uint32_t nodeGroupSizeRest = nUeNodes % numActiveGnbs;
            uint32_t ueIndexCounter = 0;

            // Aloca UEs nos gNBs selecionados
            for (uint32_t bsCoordIndex = 0; bsCoordIndex < numActiveGnbs; bsCoordIndex++)
            {
                Ptr<UniformDiscPositionAllocator> uePositionAlloc = CreateObject<UniformDiscPositionAllocator> ();
                uePositionAlloc->SetX(bsCoords[bsCoordIndex].x);
                uePositionAlloc->SetY(bsCoords[bsCoordIndex].y);
                uePositionAlloc->SetRho(isd / 2); // Raio menor em torno da BS específica
                uemobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                                            "Mode", StringValue("Time"),
                                            "Time", StringValue(std::to_string(timeDirection) + "s"),
                                            "Speed", PointerValue(speedVar),
                                            "Bounds", RectangleValue(Rectangle(0, maxXAxis, 0, maxYAxis)));
                uemobility.SetPositionAllocator(uePositionAlloc);

                uint32_t uesInThisGroup = nodeGroupSize + (bsCoordIndex < nodeGroupSizeRest ? 1 : 0);
                NodeContainer currentUeGroup;
                for(uint32_t i=0; i < uesInThisGroup && ueIndexCounter < nUeNodes; ++i)
                {
                    currentUeGroup.Add(ueNodes.Get(ueIndexCounter++));
                }
                if (currentUeGroup.GetN() > 0)
                {
                    uemobility.Install(currentUeGroup);
                    NS_LOG_INFO("Instalado " << currentUeGroup.GetN() << " UEs em torno da BS na posição " << bsCoords[bsCoordIndex]);
                }
            }
            if (ueIndexCounter != nUeNodes) {
                NS_LOG_WARN("Nem todos os UEs foram alocados em positionAllocator=1. UEs alocados: " << ueIndexCounter << ", Total UEs: " << nUeNodes);
            }
            break;
        }
        case 2: { // Mobilidade baseada em trajetórias urbanas de Shanghai
            NS_LOG_INFO("Configurando mobilidade Shanghai para " << nUeNodes << " UEs");

            // Usa a função SetupShanghaiMobility com os parâmetros do cenário
            // centerPosition é o centro do layout hexagonal
            // areaRadius é baseado no ISD para cobrir a área das células
            double areaRadius = isd * 0.9; // Raio ajustado para centralizar UEs na cobertura

            SetupShanghaiMobility(ueNodes, centerPosition.x, centerPosition.y, areaRadius, simTime);

            NS_LOG_INFO("Mobilidade Shanghai configurada: centro=(" << centerPosition.x << ","
                        << centerPosition.y << "), raio=" << areaRadius);
            break;
        }
        default:
            NS_FATAL_ERROR("positionAllocator not recognized " << positionAllocator);
            break;
    }

    // --- Configura Trace de Mobilidade ---
    mobilityTraceFile.open ("mobility-trace.txt", std::ios_base::out | std::ios_base::trunc);
    mobilityTraceFile << "#Time\tNodeType\tNodeID\tX\tY\tZ" << std::endl;

    // Agenda trace periódico para garantir dados contínuos de animação
    // Intervalo de 0.5s é bom para animação suave sem arquivo muito grande
    double traceInterval = 0.5; // segundos

    if (enableGnbMobilityTrace)
    {
        // Inclui gNBs no trace (útil para UAVs móveis)
        // NÃO usa o callback CourseChange pois ele não inclui NodeType
        Simulator::Schedule (Seconds (0.0), &PeriodicMobilityTraceWithGnbs, &mobilityTraceFile,
                             ueNodes, lteEnbNodes, mmWaveEnbNodes, traceInterval, simTime);
        NS_LOG_INFO ("Mobility trace configurado com gNBs: mobility-trace.txt (intervalo: " << traceInterval << "s)");
    }
    else
    {
        // Trace apenas UEs (comportamento original)
        // Conecta callback para mudanças de curso (formato antigo sem NodeType)
        Config::Connect ("/NodeList/*/$ns3::MobilityModel/CourseChange",
                         MakeBoundCallback (&MobilityTraceCallback, &mobilityTraceFile));
        Simulator::Schedule (Seconds (0.0), &PeriodicMobilityTrace, &mobilityTraceFile, ueNodes, traceInterval, simTime);
        NS_LOG_INFO ("Mobility trace configurado (UEs apenas): mobility-trace.txt (intervalo: " << traceInterval << "s)");
    }

    // Instala NetDevices LTE, mmWave e MC (Multi-Connectivity)
    NetDeviceContainer lteEnbDevs = mmwaveHelper->InstallLteEnbDevice (lteEnbNodes);
    NetDeviceContainer mmWaveEnbDevs = mmwaveHelper->InstallEnbDevice (mmWaveEnbNodes);
    NetDeviceContainer mcUeDevs = mmwaveHelper->InstallMcUeDevice (ueNodes);

    // Instala stack IP nas UEs e atribui endereços
    internet.Install (ueNodes);
    Ipv4InterfaceContainer ueIpIface = epcHelper->AssignUeIpv4Address (NetDeviceContainer (mcUeDevs));

    // Configura rotas default para as UEs
    for (uint32_t u = 0; u < ueNodes.GetN (); ++u)
    {
        Ptr<Node> ueNode = ueNodes.Get (u);
        Ptr<Ipv4StaticRouting> ueStaticRouting = ipv4RoutingHelper.GetStaticRouting (ueNode->GetObject<Ipv4> ());
        ueStaticRouting->SetDefaultRoute (epcHelper->GetUeDefaultGatewayAddress (), 1);
    }

    // Adiciona interface X2 entre eNBs/gNBs
    mmwaveHelper->AddX2Interface (lteEnbNodes, mmWaveEnbNodes);

    // Associa UEs à eNB/gNB mais próxima inicialmente
    mmwaveHelper->AttachToClosestEnb (mcUeDevs, mmWaveEnbDevs, lteEnbDevs);

    // --- Setup das Aplicações (tráfego - lógica do scenario-three) ---
    uint16_t portTcp = 50000;
    Address sinkLocalAddressTcp (InetSocketAddress (Ipv4Address::GetAny (), portTcp));
    PacketSinkHelper sinkHelperTcp ("ns3::TcpSocketFactory", sinkLocalAddressTcp);
    AddressValue serverAddressTcp (InetSocketAddress (remoteHostAddr, portTcp));

    uint16_t portUdp = 60000;
    Address sinkLocalAddressUdp (InetSocketAddress (Ipv4Address::GetAny (), portUdp));
    PacketSinkHelper sinkHelperUdp ("ns3::UdpSocketFactory", sinkLocalAddressUdp);
    AddressValue serverAddressUdp (InetSocketAddress (remoteHostAddr, portUdp));

    ApplicationContainer sinkApp;
    sinkApp.Add (sinkHelperTcp.Install (remoteHost)); // Sink TCP no host remoto
    sinkApp.Add (sinkHelperUdp.Install (remoteHost)); // Sink UDP no host remoto

    // Cria helpers para clientes OnOff
    OnOffHelper clientHelperTcp ("ns3::TcpSocketFactory", Address ());
    clientHelperTcp.SetAttribute ("Remote", serverAddressTcp);
    clientHelperTcp.SetAttribute ("OnTime", StringValue ("ns3::ExponentialRandomVariable[Mean=1.0]")); // Média 1s ON
    clientHelperTcp.SetAttribute ("OffTime", StringValue ("ns3::ExponentialRandomVariable[Mean=1.0]")); // Média 1s OFF
    clientHelperTcp.SetAttribute ("DataRate", StringValue (dataRate)); // Usa dataRate configurado
    clientHelperTcp.SetAttribute ("PacketSize", UintegerValue (1280));

    OnOffHelper clientHelperTcp150 ("ns3::TcpSocketFactory", Address ());
    clientHelperTcp150.SetAttribute ("Remote", serverAddressTcp);
    clientHelperTcp150.SetAttribute ("OnTime", StringValue ("ns3::ExponentialRandomVariable[Mean=1.0]"));
    clientHelperTcp150.SetAttribute ("OffTime", StringValue ("ns3::ExponentialRandomVariable[Mean=1.0]"));
    clientHelperTcp150.SetAttribute ("DataRate", StringValue ("150kbps")); // Baixa taxa
    clientHelperTcp150.SetAttribute ("PacketSize", UintegerValue (1280));

    OnOffHelper clientHelperTcp750 ("ns3::TcpSocketFactory", Address ());
    clientHelperTcp750.SetAttribute ("Remote", serverAddressTcp);
    clientHelperTcp750.SetAttribute ("OnTime", StringValue ("ns3::ExponentialRandomVariable[Mean=1.0]"));
    clientHelperTcp750.SetAttribute ("OffTime", StringValue ("ns3::ExponentialRandomVariable[Mean=1.0]"));
    clientHelperTcp750.SetAttribute ("DataRate", StringValue ("750kbps")); // Taxa média
    clientHelperTcp750.SetAttribute ("PacketSize", UintegerValue (1280));

    OnOffHelper clientHelperUdp ("ns3::UdpSocketFactory", Address ());
    clientHelperUdp.SetAttribute ("Remote", serverAddressUdp);
    clientHelperUdp.SetAttribute ("OnTime", StringValue ("ns3::ExponentialRandomVariable[Mean=1.0]"));
    clientHelperUdp.SetAttribute ("OffTime", StringValue ("ns3::ExponentialRandomVariable[Mean=1.0]"));
    clientHelperUdp.SetAttribute ("DataRate", StringValue (dataRate));
    clientHelperUdp.SetAttribute ("PacketSize", UintegerValue (1280));

    ApplicationContainer clientApp;
    switch (trafficModel)
    {
        case 0: { // Full Buffer (constante UDP)
            for (uint32_t u = 0; u < ueNodes.GetN (); ++u)
            {
                PacketSinkHelper dlPacketSinkHelper ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), 1234));
                sinkApp.Add (dlPacketSinkHelper.Install (ueNodes.Get (u))); // Sink na UE
                UdpClientHelper dlClient (ueIpIface.GetAddress (u), 1234); // Cliente no Host Remoto
                dlClient.SetAttribute ("MaxPackets", UintegerValue (UINT32_MAX));
                dlClient.SetAttribute ("PacketSize", UintegerValue (1280));
                // Calcula intervalo para dataRate desejado (aproximado)
                DataRate targetRate(dataRate);
                Time pktInterval = Seconds(1280.0 * 8.0 / targetRate.GetBitRate());
                dlClient.SetAttribute ("Interval", TimeValue (pktInterval));
                clientApp.Add (dlClient.Install (remoteHost));
            }
        }
        break;

        case 1: { // Metade Full Buffer, Metade Bursty (OnOff)
            for (uint32_t u = 0; u < ueNodes.GetN (); ++u)
            {
                if (u % 2 == 0) // Bursty
                {
                    PacketSinkHelper dlPacketSinkHelper ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), 1234));
                    sinkApp.Add (dlPacketSinkHelper.Install (ueNodes.Get (u))); // Sink na UE
                    clientApp.Add (clientHelperUdp.Install (ueNodes.Get(u))); // Cliente OnOff UDP na UE
                }
                else // Full Buffer
                {
                    PacketSinkHelper dlPacketSinkHelper ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), 1234));
                    sinkApp.Add (dlPacketSinkHelper.Install (ueNodes.Get (u))); // Sink na UE
                    UdpClientHelper dlClient (ueIpIface.GetAddress (u), 1234); // Cliente no Host Remoto
                    dlClient.SetAttribute ("MaxPackets", UintegerValue (UINT32_MAX));
                    dlClient.SetAttribute ("PacketSize", UintegerValue (1280));
                    DataRate targetRate(dataRate);
                    Time pktInterval = Seconds(1280.0 * 8.0 / targetRate.GetBitRate());
                    dlClient.SetAttribute ("Interval", TimeValue (pktInterval));
                    clientApp.Add (dlClient.Install (remoteHost));
                }
            }
        }
        break;

        case 2: { // Tudo Bursty (OnOff)
            for (uint32_t u = 0; u < ueNodes.GetN (); ++u)
            {
                // Instala Sink TCP e UDP na UE
                PacketSinkHelper dlPacketSinkTcp ("ns3::TcpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), 1235));
                PacketSinkHelper dlPacketSinkUdp ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), 1234));
                sinkApp.Add(dlPacketSinkTcp.Install(ueNodes.Get(u)));
                sinkApp.Add(dlPacketSinkUdp.Install(ueNodes.Get(u)));

                // Alterna entre Cliente TCP e UDP na UE
                if (u % 2 == 0)
                {
                    // Configura cliente TCP para enviar para o sink TCP na UE
                    clientHelperTcp.SetAttribute("Remote", AddressValue(InetSocketAddress(ueIpIface.GetAddress(u), 1235)));
                    clientApp.Add (clientHelperTcp.Install (ueNodes.Get (u)));
                }
                else
                {
                    // Configura cliente UDP para enviar para o sink UDP na UE
                    clientHelperUdp.SetAttribute("Remote", AddressValue(InetSocketAddress(ueIpIface.GetAddress(u), 1234)));
                    clientApp.Add (clientHelperUdp.Install (ueNodes.Get (u)));
                }
            }
        }
        break;

        case 3: {
            // Text: "mixture of four heterogeneous traffic models"
            for (uint32_t u = 0; u < ueNodes.GetN (); ++u)
            {
                // Instala Sinks (Receptores) na UE para TCP e UDP
                PacketSinkHelper dlPacketSinkTcp ("ns3::TcpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), 1235));
                sinkApp.Add(dlPacketSinkTcp.Install(ueNodes.Get(u)));
                PacketSinkHelper dlPacketSinkUdp ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), 1234));
                sinkApp.Add (dlPacketSinkUdp.Install (ueNodes.Get (u)));

                // Destination Address (UE)
                AddressValue ueSinkAddrTcp(InetSocketAddress(ueIpIface.GetAddress(u), 1235));
                AddressValue ueSinkAddrUdp(InetSocketAddress(ueIpIface.GetAddress(u), 1234));

                if (u % 4 == 0) // 25%: TCP full-buffer, 20 Mbps
                {
                    // Note: Text says "TCP full-buffer... with a data rate of 20 Mbps".
                    // We use OnOff with TCP, High OnTime, and specific DataRate to limit to 20Mbps.
                    OnOffHelper client = clientHelperTcp;
                    client.SetAttribute("Remote", ueSinkAddrTcp);
                    client.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=100.0]")); // Always ON
                    client.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0.0]"));
                    client.SetAttribute("DataRate", StringValue("20Mbps"));
                    clientApp.Add(client.Install(remoteHost));
                }
                else if (u % 4 == 1) // 25%: UDP bursty, Avg 20 Mbps
                {
                    // Text: "UDP bursty... averaging around 20 Mbps"
                    // Assuming Exp(1.0) for ON and OFF, Duty Cycle is 50%.
                    // Peak rate must be 40Mbps to average 20Mbps.
                    OnOffHelper client = clientHelperUdp;
                    client.SetAttribute("Remote", ueSinkAddrUdp);
                    client.SetAttribute("OnTime", StringValue ("ns3::ExponentialRandomVariable[Mean=1.0]"));
                    client.SetAttribute("OffTime", StringValue ("ns3::ExponentialRandomVariable[Mean=1.0]"));
                    client.SetAttribute("DataRate", StringValue("40Mbps"));
                    clientApp.Add(client.Install(remoteHost));
                }
                else if (u % 4 == 2) // 25%: TCP bursty, Avg 750 kbps
                {
                    // Text: "TCP bursty... averaging 750 kb/s"
                    // Peak rate = 1.5 Mbps (50% duty cycle)
                    OnOffHelper client = clientHelperTcp;
                    client.SetAttribute("Remote", ueSinkAddrTcp);
                    client.SetAttribute("OnTime", StringValue ("ns3::ExponentialRandomVariable[Mean=1.0]"));
                    client.SetAttribute("OffTime", StringValue ("ns3::ExponentialRandomVariable[Mean=1.0]"));
                    client.SetAttribute("DataRate", StringValue("1.5Mbps"));
                    clientApp.Add(client.Install(remoteHost));
                }
                else if (u % 4 == 3) // 25%: TCP bursty, Avg 150 kbps
                {
                    // Text: "TCP bursty... averaging 150 kbps"
                    // Peak rate = 300 kbps (50% duty cycle)
                    OnOffHelper client = clientHelperTcp;
                    client.SetAttribute("Remote", ueSinkAddrTcp);
                    client.SetAttribute("OnTime", StringValue ("ns3::ExponentialRandomVariable[Mean=1.0]"));
                    client.SetAttribute("OffTime", StringValue ("ns3::ExponentialRandomVariable[Mean=1.0]"));
                    client.SetAttribute("DataRate", StringValue("300kbps"));
                    clientApp.Add(client.Install(remoteHost));
                }
            }
            break;
        }

        default:
            NS_FATAL_ERROR ( "Modelo de tráfego inválido: " << trafficModel);
    }


    // --- Início e Fim da Simulação ---
    // Nota: simTime já foi lido anteriormente para uso na mobilidade Shanghai
    sinkApp.Start (Seconds (0.1)); // Pequeno atraso para garantir que os sinks estão prontos
    clientApp.Start (Seconds (0.2)); // Pequeno atraso para iniciar clientes
    clientApp.Stop (Seconds (simTime - 0.1)); // Para um pouco antes do fim

    // Ativa traces se necessário
    if (enableTraces)
    {
        mmwaveHelper->EnableTraces ();
    }

    // Ativa traces LTE PHY/MAC
    Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();
    lteHelper->Initialize ();
    lteHelper->EnablePhyTraces ();
    lteHelper->EnableMacTraces ();

    // Imprime posições iniciais para gnuplot
    PrintGnuplottableUeListToFile ("ues.txt");
    PrintGnuplottableEnbListToFile ("enbs.txt");

    // --- Agendamento do Log de Estado da BS (BsStateTrace) ---
    // Obtém o NetDevice e RRC do eNB LTE (assumindo que só há um)
    Ptr<LteEnbNetDevice> ltedev = DynamicCast<LteEnbNetDevice> (lteEnbDevs.Get (0));
    if (!ltedev) {
        NS_FATAL_ERROR("Não foi possível encontrar o LteEnbNetDevice.");
    }
    Ptr<LteEnbRrc> lte_rrc = ltedev->GetRrc ();
    if (!lte_rrc) {
        NS_FATAL_ERROR("Não foi possível obter o LteEnbRrc.");
    }

    if (pathGymOkMetrics)
    {
        // path-gym-ok reversible hook: keep extra exports behind one scenario flag.
        if (!enableTraces)
        {
            mmwaveHelper->EnableRlcTraces ();
            mmwaveHelper->EnablePdcpTraces ();
        }

        lteHelper->EnableRlcTraces ();
        lteHelper->EnablePdcpTraces ();

        Config::ConnectFailSafe (
            "/NodeList/*/DeviceList/*/ComponentCarrierMapUe/*/LteUePhy/ReportCurrentCellRsrpSinr",
            MakeCallback (&PathGymOkLteSinrTrace));
        Config::ConnectFailSafe (
            "/NodeList/*/DeviceList/*/LteComponentCarrierMapUe/*/LteUePhy/ReportCurrentCellRsrpSinr",
            MakeCallback (&PathGymOkLteSinrTrace));
        Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/LteEnbRrc/NotifyMmWaveSinr",
                                 MakeCallback (&PathGymOkMmWaveSinrTrace));

        Simulator::Schedule (Seconds (0.0),
                             &PathGymOkSampleMetrics,
                             ltedev,
                             mmWaveEnbDevs,
                             outageThreshold,
                             indicationPeriodicity,
                             simTime);
    }

    // Agenda a escrita do estado a cada `indicationPeriodicity`
    int numSteps = static_cast<int>(std::ceil(simTime / indicationPeriodicity));
    for (int step = 0; step <= numSteps; ++step) {
        double time = step * indicationPeriodicity;
        if (time <= simTime + 0.0001) {
            Simulator::Schedule(Seconds(time), BsStateTrace, "bsState.txt", ltedev, lte_rrc);
        }
    }

    // --- Configuração de Heurísticas Energy Saving ---
    NS_LOG_INFO ("Configuring Energy Saving heuristic: type=" << (int)heuristicType);

    Ptr<EnergyHeuristic> energyHeur = CreateObject<EnergyHeuristic> ();

    switch (heuristicType)
    {
        case -1: {
            // No heuristic - External control via semaphores/file
            NS_LOG_INFO ("No heuristic - External RL control enabled");
            // Células começam todas ON, controle externo decide
            break;
        }
        case 0: {
            // Always ON - All cells stay ON
            NS_LOG_INFO ("Always ON heuristic - All cells will remain ON");
            // Todas as células já começam ON por padrão, nada a fazer
            break;
        }
        case 1: {
            // Dynamic sleeping based on SINR (como scenario-three.cc)
            NS_LOG_INFO ("Dynamic sleeping heuristic enabled with sinrTh=" << sinrTh);
            NS_LOG_INFO ("BS config: bsOn=" << bsOn << " bsIdle=" << bsIdle
                         << " bsSleep=" << bsSleep << " bsOff=" << bsOff);

            // BsStatus array: [bsOn, bsIdle, bsSleep, bsOff]
            int BsStatus[4] = {bsOn, bsIdle, bsSleep, bsOff};

            // Verifica que a soma é igual ao número de células mmWave
            if (bsOn + bsIdle + bsSleep + bsOff != nMmWaveEnbNodes)
            {
                NS_LOG_WARN ("Warning: bsOn+bsIdle+bsSleep+bsOff=" << (bsOn + bsIdle + bsSleep + bsOff)
                             << " differs from nMmWaveEnbNodes=" << (int)nMmWaveEnbNodes
                             << ". Adjusting bsOn to match.");
                // Ajusta bsOn para completar a soma
                BsStatus[0] = nMmWaveEnbNodes - bsIdle - bsSleep - bsOff;
                if (BsStatus[0] < 0) BsStatus[0] = 0;
            }

            // Se bsIdle == 0, trata bsIdle como bsOn (permite células OFF ligarem)
            if (bsIdle == 0)
            {
                BsStatus[1] = BsStatus[0];  // bsIdle = bsOn
                BsStatus[0] = 0;            // bsOn = 0
            }

            for (double t = indicationPeriodicity; t <= simTime; t += indicationPeriodicity)
            {
                // Se bsOn==0, não precisa contar SINR dos UEs conectados
                for (int j = 0; j < nMmWaveEnbNodes && BsStatus[0] != 0; j++)
                {
                    Ptr<MmWaveEnbNetDevice> mmdev = DynamicCast<MmWaveEnbNetDevice> (mmWaveEnbDevs.Get (j));
                    Simulator::Schedule (Seconds (t), &EnergyHeuristic::CountBestUesSinr, energyHeur, sinrTh, mmdev);
                }
                Simulator::Schedule (Seconds (t), &EnergyHeuristic::TurnOnBsSinrPos, energyHeur,
                                     nMmWaveEnbNodes, mmWaveEnbDevs, std::string("dynamic"), BsStatus, ltedev);
            }
            break;
        }
        default: {
            NS_LOG_WARN ("Unknown heuristicType: " << (int)heuristicType << ". Using no heuristic.");
            break;
        }
    }

    // Mensagem de início e execução da simulação
    NS_LOG_UNCOND ("Hierarchical Simulation Starting. Time: " << simTime
                   << " seconds. Control File: '" << controlFilename
                   << "' Use Semaphores: " << useSemaphores
                   << " Schedule Control Messages: " << scheduleControlMessages);
    NS_LOG_UNCOND ("Heuristic Type: " << (int)heuristicType << " (-1=RL, 0=AlwaysON, 1=Dynamic)");
    NS_LOG_UNCOND ("UAV Configuration: mode=" << (uavMobilityMode == 0 ? "static" : "mobile")
                   << ", pattern=" << (int)uavFlightPattern << ", altitude=" << uavBaseAltitude << "m"
                   << ", speed=" << uavMaxSpeed << "m/s");
    Simulator::Stop (Seconds (simTime));
    Simulator::Run ();

    // Fecha arquivos de trace
    if (mobilityTraceFile.is_open ())
    {
        mobilityTraceFile.close ();
        NS_LOG_INFO ("Mobility trace salvo em: mobility-trace.txt");
    }
    if (outFile.is_open ())
    {
        outFile.close ();
    }

    Simulator::Destroy ();
    NS_LOG_INFO ("Done.");
    return 0;
}
